#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
import re

def removePunc(str):
    # remove all tokens that are not alphabetic
    res = ""
    content = str.split(" ")
    for word in content:
        if word.isalpha():
            res += f" {word} "
        else:
            cleanr = re.compile(r"(\.|'|\?|!|\"|,)")
            cleantext = re.sub(cleanr, '', word)
            res += f"{ cleantext} "
    return res

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    text_string = ""
    if len(content) > 1:
        ### remove punctuation
        # Removing all non-alphabetic words

        for each in content:
            text_string += removePunc(each)




        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)

        word_arr = text_string.split(" ")


        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        res = []
        for each in word_arr:
            # print(stemmer.stem(each))
            res.append(stemmer.stem(each))

        # print(res)
        for each in res:
            words += f" {each}"

    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()

