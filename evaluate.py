import torch
import random
from train import indexesFromSentence
from dataloader import SOS_token, EOS_token
from dataloader import MAX_LENGTH, loadPrepareData, Voc, normalizeString
from model import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")



class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores



class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())

def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for i in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]])
            decoder_input = decoder_input.to(device)

            decoder_hidden = sentence.decoder_hidden
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]

def decode(decoder, decoder_hidden, encoder_outputs, voc, max_length=MAX_LENGTH):

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length) #TODO: or (MAX_LEN+1, MAX_LEN+1)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        _, topi = decoder_output.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[ni.item()])

        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.to(device)

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, voc, sentence, beam_size, max_length=MAX_LENGTH, searcher=None):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
    # Create lengths tensor
    lengths = [len(indexes) for indexes in indexes_batch]
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    # lengths = lengths.to("cpu")

    if searcher:
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, max_length)
        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    else:
        encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)

        decoder_hidden = encoder_hidden[:decoder.n_layers]

        if beam_size == 1:
            return decode(decoder, decoder_hidden, encoder_outputs, voc)
        else:
            return beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size)


def evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, n=10, seacher=None):
    for _ in range(n):
        pair = random.choice(pairs)
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', pair[0])
        
        if searcher:
            # Normalize sentence
            input_sentence = normalizeString(pair)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, voc, input_sentence, beam_size=None, searcher=searcher)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))
        
        else:
            if beam_size == 1:
                output_words, _ = evaluate(encoder, decoder, voc, pair[0], beam_size)
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
            else:
                output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))
            


def evaluateInput(encoder, decoder, voc, beam_size, searcher=None):
    pair = ''
    while(1):
        try:
            # Get input
            pair = input('> ')
             # Check if it is quit case
            if pair == 'q' or pair == 'quit': break

            if searcher:
                # Normalize sentence
                input_sentence = normalizeString(pair)
                # Evaluate sentence
                output_words = evaluate(encoder, decoder, voc, input_sentence, beam_size=None, searcher=searcher)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))
            
            else:
                if beam_size == 1:
                    output_words, _ = evaluate(encoder, decoder, voc, pair, beam_size)
                    output_sentence = ' '.join(output_words)
                    print('<', output_sentence)
                else:
                    output_words_list = evaluate(encoder, decoder, voc, pair, beam_size)
                    for output_words, score in output_words_list:
                        output_sentence = ' '.join(output_words)
                        print("{:.3f} < {}".format(score, output_sentence))
        except KeyError:
            print("Incorrect spelling.")


def runTest_BeamSearch(n_layers, hidden_size, reverse, modelFile, beam_size, inp, corpus):
    torch.set_grad_enabled(False)

    voc, pairs = loadPrepareData(corpus)
    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(voc.num_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, n_layers)

    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if inp:
        evaluateInput(encoder, decoder, voc, beam_size)
    else:
        evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, n=20)


def runTest_GreedySearch(n_layers, hidden_size, modelFile, inp, corpus, beam_size=None, reverse=None):
    voc, pairs = loadPrepareData(corpus)
    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(voc.num_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, n_layers)
    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    searcher = GreedySearchDecoder(encoder, decoder)

    if inp:
        evaluateInput(encoder, decoder, voc, beam_size=None, searcher=searcher)
    else:
        evaluateRandomly(encoder, decoder, voc, pairs, reverse=None, beam_size=None, n=20, searcher=searcher)
