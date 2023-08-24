import json
import os
import csv
import codecs

data_dir = 'data'
processed_dir = os.path.join(data_dir, 'processed')

inputFile = os.path.join(data_dir, 'movie-corpus', 'utterances.jsonl')
outputFile = os.path.join(processed_dir, 'formatted_movie_lines.txt')

def splitLinesConversations(fileName):

    # ... (paste your existing splitLinesConversations function here)
    lines={}
    conversations={}
    with open(fileName, 'r', encoding='iso-8859-1') as f:       # ISO/IEC 8859-1 encodes what it refers to as "Latin alphabet no. 1",
        for line in f:                                          # consisting of 191 characters from the Latin script.
            lineJson = json.loads(line)                         # This character-encoding scheme is used throughout the Americas,
                                                                # Western Europe, Oceania, and much of Africa.
            # Extract fields for line object
            lineObj = {}
            lineObj["lineID"] = lineJson["id"]
            lineObj["characterID"] = lineJson["speaker"]
            lineObj["text"] = lineJson["text"]
            lines[lineObj["lineID"]] = lineObj

            # Extract fields for conversation object
            if lineJson["conversation_id"] not in conversations:
                convObj = {}
                convObj["conversationID"] = lineJson["conversation_id"]
                convObj["movieID"] = lineJson["meta"]["movie_id"]
                convObj["lines"] = [lineObj]
            else:
                convObj = conversations[lineJson["conversation_id"]]
                convObj["lines"].insert(0, lineObj)
            conversations[convObj["conversationID"]] = convObj

    return lines, conversations

def extractSentencePairs(conversations):
    # ... (paste your existing extractSentencePairs function here)
    qa_pairs=[]
    for conversation in conversations.values():
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])

    return qa_pairs

def writeFile(inputfile, outputfile):
    print("Formatting file....")
    lines = {}
    conversations = {}
    lines, conversations = splitLinesConversations(inputfile)
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    with open(outputfile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)