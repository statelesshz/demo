import codecs
import csv

from pipeline import pipeline


def chat_piepline_with_only_task():
    chat_pipeline = pipeline("chat")
    print(chat_pipeline("hi"))


if __name__ == "__main__":
    chat_piepline_with_only_task()
    