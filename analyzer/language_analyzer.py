from enum import Enum, IntEnum

import nltk
import URLExtract
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from transformers import pipeline


class NeuralNetRuntime(IntEnum):
    USE_GPU = 0,
    USE_CPU = 1


class LanguageAnalyzer(object):
    # Due to UTF-8, we have to deal with variable length encoding which doesnt allow you to accurately calculate tensor
    # length before hand so its basically 2056 - random stuff
    MAX_TENSOR_SIZE = 1750

    def __init__(self, should_perform_zero_shot_analysis=True, nn_runtime=NeuralNetRuntime.USE_GPU):
        nltk.download('stopwords')  # I do this so you dont have 2!
        self.punctuation_tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english') + (stopwords.words('norwegian')))

        nltk.download('vader_lexicon')
        self.candidate_labels = ['positiv', 'negativ']
        self.hypothesis_template = 'Dette eksempelet er {}.'
        self.extractor = URLExtract()
        if should_perform_zero_shot_analysis:
            self.classifier = pipeline("zero-shot-classification",
                                       model="NbAiLab/nb-bert-base-mnli",
                                       device=nn_runtime.value)  # This takes 24 GB GPU memory so only run if you have a 3090 or better otherwise dont set device to 0, you need more than 24 gb ram tho

    @staticmethod
    def avg(lst):
        return sum(lst) / len(lst)

    # This should never be done, it should have an <inaccurate> flag next to it or smth
    def extract_multi_length_tensor_sentiment(self, classification_result):
        retval = {
            "positiv": 0,
            "negativ": 0
        }
        positive_list = []
        negative_list = []
        for classifier in classification_result:
            try:
                if classifier['labels'][0] == "positiv":
                    positive_list.append(classifier['scores'][0])
                    negative_list.append(classifier['scores'][1])
                else:
                    positive_list.append(classifier['scores'][1])
                    negative_list.append(classifier['scores'][0])
            except TypeError:
                print(classification_result)  # WTF?
                return retval
        retval["positiv"] = self.avg(positive_list)
        retval["negativ"] = self.avg(negative_list)
        return retval

    def try_sentiment_analysis_for_comment_reduce_tensor(self, comment, previous_tensor_size):
        reduced_tensor_size_due_to_misaligned_utf_8 = previous_tensor_size - 256
        chunks = [comment[i:i + previous_tensor_size] for i in range(0, len(comment), previous_tensor_size)]
        try:
            classification_result = self.classifier(chunks, self.candidate_labels,
                                                    hypothesis_template=self.hypothesis_template, multi_label=True)
        except RuntimeError as e:
            return self.try_sentiment_analysis_for_comment_reduce_tensor(comment,
                                                                         reduced_tensor_size_due_to_misaligned_utf_8)
        return self.extract_multi_length_tensor_sentiment(classification_result)

    def perform_sentiment_analysis_for_comment(self, comment):
        retval = {
            "positiv": 0,
            "negativ": 0
        }
        if comment == "":
            return retval
        if len(comment) >= self.MAX_TENSOR_SIZE:
            chunks = [comment[i:i + self.MAX_TENSOR_SIZE] for i in range(0, len(comment),self.MAX_TENSOR_SIZE)]
            try:
                classification_result = self.classifier(chunks, self.candidate_labels,
                                                        hypothesis_template=self.hypothesis_template, multi_label=True)
            except RuntimeError:
                return self.try_sentiment_analysis_for_comment_reduce_tensor(comment, (self.MAX_TENSOR_SIZE - 512))
            return self.extract_multi_length_tensor_sentiment(classification_result)

        try:
            classification_result = self.classifier(comment, self.candidate_labels,
                                                    hypothesis_template=self.hypothesis_template, multi_label=True)
        except Exception as e:
            print(e)
            return retval
        self.create_sentiment_scores(classification_result, retval)
        return retval

    def create_sentiment_scores(self, classification_result, retval):
        if classification_result['labels'][0] == "positiv":
            retval["positiv"] = classification_result['scores'][0]
            retval["negativ"] = classification_result['scores'][1]
        else:
            retval["positiv"] = classification_result['scores'][1]
            retval["negativ"] = classification_result['scores'][0]

    def clean_and_create_text_for_analysis(self, comment):
        """
        This methods remove all punctuations (.,!?), special tokens ("#¤%&/()= etc) and all stopwords in both Norwegian,
            and english
        :param comment: Comment and/or post text ready for analysis
        :return: Cleaned comment without any stopwords etc
        """
        urls = self.extractor.find_urls(comment)
        for url in urls:
            comment.replace(url, "")
        cleaned_comment = self.punctuation_tokenizer.tokenize(comment)
        cleaned_comment = [word for word in cleaned_comment if not word in self.stop_words]
        return " ".join(cleaned_comment)
