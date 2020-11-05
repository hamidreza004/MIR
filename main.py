import tkinter
from tkinter import *
from tkinter import filedialog
import preprocess.preprocess_eng as eng
import preprocess.preprocess_per as per
import tkinter as tk
import pandas as pd
from helper import XML_to_dataframe
from index.core import add_single_document, add_multiple_documents


class EntryWithPlaceholder(tk.Entry):
    def __init__(self, master=None, placeholder="PLACEHOLDER", color='grey'):
        super().__init__(master)

        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['fg']

        self.bind("<FocusIn>", self.foc_in)
        self.bind("<FocusOut>", self.foc_out)

        self.put_placeholder()

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color

    def foc_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete('0', 'end')
            self['fg'] = self.default_fg_color

    def foc_out(self, *args):
        if not self.get():
            self.put_placeholder()


def wide_table(list, number_of_rows):
    h = len(list) * number_of_rows
    w = (len(list[0]) + number_of_rows - 1) // number_of_rows
    res = [["" for x in range(w)] for y in range(h)]
    for i in range(len(list)):
        for j in range(len(list[0])):
            res[(j % number_of_rows) * len(list) + i][j // number_of_rows] = list[i][j]
    return res


def add_table(window, list):
    for i in range(len(list)):
        window.grid_columnconfigure(i, weight=1)
    for i in range(len(list[0])):
        window.grid_rowconfigure(i, weight=1)
    for i in range(len(list[0])):
        for j in range(len(list)):
            e = Entry(window)
            e.grid(row=i, column=j, padx=0, pady=0, ipadx=0, ipady=0, sticky=W + E + N + S)
            e.insert(END, list[j][i])
            e.config(state="readonly")


def configure_size_window(window):
    window.title("MIR Project")
    window.geometry('840x500')
    window.grid_columnconfigure(1, weight=1)
    window.grid_columnconfigure(2, weight=1)
    window.grid_columnconfigure(3, weight=1)
    window.grid_rowconfigure(0, weight=1)
    window.grid_rowconfigure(1, weight=1)
    window.grid_rowconfigure(2, weight=1)
    window.grid_rowconfigure(3, weight=1)
    window.grid_rowconfigure(4, weight=1)
    window.grid_rowconfigure(5, weight=1)
    window.grid_rowconfigure(6, weight=1)
    window.grid_rowconfigure(7, weight=1)
    window.grid_rowconfigure(8, weight=1)


def configure_prepare_section(window):
    def prepare_CSV_clicked():
        filename = filedialog.askopenfilename()
        df = pd.read_csv(filename)
        df = df[['description', 'title']]
        result_df, stop_words = eng.prepare_text(df)
        stopwords_window = Toplevel(window)
        stopwords_window.title("Stopwords found (TOP {}%)".format(eng.stop_word_ratio * 100))
        stopwords_window.geometry("800x800")
        add_table(stopwords_window, wide_table(stop_words, 6))
        print(df)
        stopwords_window.mainloop()

    btn_CSV = Button(window, text="Prepare CSV documents", command=prepare_CSV_clicked)
    btn_CSV.grid(column=1, row=0, sticky=W + E + N + S, columnspan=2)

    def prepare_XML_clicked():
        filename = filedialog.askopenfilename()
        df = XML_to_dataframe(filename)
        df = df[['description', 'title']]
        result_df, stop_words = per.prepare_text(df)
        stopwords_window = Toplevel(window)
        stopwords_window.title("Stopwords found (TOP {}%)".format(per.stop_word_ratio * 100))
        stopwords_window.geometry("800x1200")
        add_table(stopwords_window, wide_table(stop_words, 6))
        print(df)
        stopwords_window.mainloop()

    btn_XML = Button(window, text="Prepare XML documents", command=prepare_XML_clicked)
    btn_XML.grid(column=3, row=0, sticky=W + E + N + S, columnspan=1)


def initial_window(window):
    configure_size_window(window)
    configure_prepare_section(window)
    entry_delete_doc = EntryWithPlaceholder(window, "Enter document text, Hello! etc.")
    entry_delete_doc.grid(column=1, row=1, sticky=W + E + N + S, columnspan=1)
    entry_delete_doc = EntryWithPlaceholder(window, "Enter document title, Intro etc.")
    entry_delete_doc.grid(column=2, row=1, sticky=W + E + N + S, columnspan=1)
    btn_delete_doc = Button(window, text="Add single document")
    btn_delete_doc.grid(column=3, row=1, sticky=W + E + N + S, columnspan=1)
    entry_delete_doc = EntryWithPlaceholder(window, "Enter document ID, 32198 etc.")
    entry_delete_doc.grid(column=1, row=2, sticky=W + E + N + S, columnspan=2)
    btn_delete_doc = Button(window, text="Delete single document")
    btn_delete_doc.grid(column=3, row=2, sticky=W + E + N + S)
    entry_posting_list = EntryWithPlaceholder(window, "Enter term, Hello etc.")
    entry_posting_list.grid(column=1, row=3, sticky=W + E + N + S, columnspan=2)
    btn_show_posting_list = Button(window, text="Show posting-list of a term")
    btn_show_posting_list.grid(column=3, row=3, sticky=W + E + N + S)
    entry_term_pos = EntryWithPlaceholder(window, "Enter term, Hello etc.")
    entry_term_pos.grid(column=1, row=4, sticky=W + E + N + S, columnspan=1)
    entry_doc_pos = EntryWithPlaceholder(window, "Enter document ID, 32198 etc.")
    entry_doc_pos.grid(column=2, row=4, sticky=W + E + N + S, columnspan=1)
    btn_show_pos_term_doc = Button(window, text="Show positions of term in document")
    btn_show_pos_term_doc.grid(column=3, row=4, sticky=W + E + N + S)
    entry_bigram_terms = EntryWithPlaceholder(window, "Enter bigram terms, ba-ac-dv-ef etc.")
    entry_bigram_terms.grid(column=1, row=5, sticky=W + E + N + S, columnspan=2)
    btn_show_term_bigram = Button(window, text="Show terms fit in this bigram")
    btn_show_term_bigram.grid(column=3, row=5, sticky=W + E + N + S)
    btn_save = Button(window, text="Save index")
    btn_save.grid(column=1, row=6, sticky=W + E + N + S, columnspan=3)
    entry_query = EntryWithPlaceholder(window, "Enter your query, Shakespeare book etc.")
    entry_query.grid(column=1, row=7, sticky=W + E + N + S, columnspan=3)
    btn_correct = Button(window, text="Correct my query")
    btn_correct.grid(column=1, row=8, sticky=W + E + N + S, columnspan=1)
    btn_search_lnc = Button(window, text="LNC-LTC search")
    btn_search_lnc.grid(column=2, row=8, sticky=W + E + N + S, columnspan=1)
    btn_search_prox = Button(window, text="Proximity search")
    btn_search_prox.grid(column=3, row=8, sticky=W + E + N + S, columnspan=1)


window = tkinter.Tk()
initial_window(window)
window.mainloop()
