import tkinter
from tkinter import *
from tkinter import filedialog
import preprocess.preprocess_eng as eng
import preprocess.preprocess_per as per
import tkinter as tk
import pandas as pd
from helper import XML_to_dataframe
import index.core as index


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
        index.add_multiple_documents(df)
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
        index.add_multiple_documents(df)
        stopwords_window.mainloop()

    btn_XML = Button(window, text="Prepare XML documents", command=prepare_XML_clicked)
    btn_XML.grid(column=3, row=0, sticky=W + E + N + S, columnspan=1)


def configure_change_index_section(window):
    entry_add_doc_desc = EntryWithPlaceholder(window, "Enter document desc, Hello! etc.")
    entry_add_doc_desc.grid(column=1, row=1, sticky=W + E + N + S, columnspan=1)
    entry_add_doc_title = EntryWithPlaceholder(window, "Enter document title, Intro etc.")
    entry_add_doc_title.grid(column=2, row=1, sticky=W + E + N + S, columnspan=1)
    btn_add_doc = Button(window, text="Add single document")
    btn_add_doc.grid(column=3, row=1, sticky=W + E + N + S, columnspan=1)

    entry_delete_doc = EntryWithPlaceholder(window, "Enter document ID, 32198 etc.")
    entry_delete_doc.grid(column=1, row=2, sticky=W + E + N + S, columnspan=2)
    btn_delete_doc = Button(window, text="Delete single document")
    btn_delete_doc.grid(column=3, row=2, sticky=W + E + N + S)


def configure_index_section(window):
    configure_change_index_section(window)
    entry_posting_list = EntryWithPlaceholder(window, "Enter term, Hello etc.")
    entry_posting_list.grid(column=1, row=3, sticky=W + E + N + S, columnspan=2)

    def show_posting_list_clicked():
        token = entry_posting_list.get()
        posting_list = []
        if index.token_exists(token):
            posting_list = index.positional[index.get_token_id(token)]
        posting_list_window = Toplevel(window)
        posting_list_window.title("Posting list of  '{}'".format(token))
        posting_list_window.geometry("450x450")
        scrollbar = Scrollbar(posting_list_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox = Listbox(posting_list_window, yscrollcommand=scrollbar.set)
        for doc_pos in posting_list:
            listbox.insert(END, "Document {}:".format(doc_pos[0]))
            for position in doc_pos[1]:
                if position % 2 == 0:
                    listbox.insert(END, "Position {} of title".format(position // 2))
                else:
                    listbox.insert(END, "Position {} of description".format(position // 2))
        listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        posting_list_window.mainloop()

    btn_show_posting_list = Button(window, text="Show posting-list of a term", command=show_posting_list_clicked)
    btn_show_posting_list.grid(column=3, row=3, sticky=W + E + N + S)

    entry_term_pos = EntryWithPlaceholder(window, "Enter term, Hello etc.")
    entry_term_pos.grid(column=1, row=4, sticky=W + E + N + S, columnspan=1)
    entry_doc_pos = EntryWithPlaceholder(window, "Enter document ID, 32198 etc.")
    entry_doc_pos.grid(column=2, row=4, sticky=W + E + N + S, columnspan=1)

    def show_pos_term_clicked():
        token = entry_term_pos.get()
        document = entry_doc_pos.get()
        positions = []
        if index.token_exists(token):
            for doc_pos in index.positional[index.get_token_id(token)]:
                if str(doc_pos[0]) == document:
                    positions = doc_pos[1]
                    break
        postition_list_window = Toplevel(window)
        postition_list_window.title("Positions '{}' appeared in document '{}'".format(token, document))
        postition_list_window.geometry("450x400")
        scrollbar = Scrollbar(postition_list_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox = Listbox(postition_list_window, yscrollcommand=scrollbar.set)
        for position in positions:
            if position % 2 == 0:
                listbox.insert(END, "Position {} of title".format(position // 2))
            else:
                listbox.insert(END, "Position {} of description".format(position // 2))
        listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        postition_list_window.mainloop()

    btn_show_pos_term_doc = Button(window, text="Show positions of term in document", command=show_pos_term_clicked)
    btn_show_pos_term_doc.grid(column=3, row=4, sticky=W + E + N + S)

    entry_bigram_terms = EntryWithPlaceholder(window, "Enter bigram terms, ba etc.")
    entry_bigram_terms.grid(column=1, row=5, sticky=W + E + N + S, columnspan=2)

    def show_bigram_list_clicked():
        biword = entry_bigram_terms.get()
        tokens = []
        if biword in index.bigram:
            for token_id in index.bigram[biword]:
                tokens.append(index.all_tokens[token_id])
        token_list_window = Toplevel(window)
        token_list_window.title("Terms contain '{}'".format(biword))
        token_list_window.geometry("300x500")
        scrollbar = Scrollbar(token_list_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox = Listbox(token_list_window, yscrollcommand=scrollbar.set)
        for token in tokens:
            listbox.insert(END, token)
        listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        token_list_window.mainloop()

    btn_show_term_bigram = Button(window, text="Show terms fit in this bigram", command=show_bigram_list_clicked)
    btn_show_term_bigram.grid(column=3, row=5, sticky=W + E + N + S)


def initial_window(window):
    configure_size_window(window)
    configure_prepare_section(window)
    configure_index_section(window)
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
