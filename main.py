import tkinter
from tkinter import *
from tkinter import filedialog
import preprocess.preprocess_eng as eng
import preprocess.preprocess_per as per
import preprocess.stopwords as stopwords_core
import tkinter as tk
import pandas as pd
from helper import XML_to_dataframe
import index.core as index
import index.spell_checker as spell_checker
import search.LNC_LTC as LNC_LTC
import search.proximity as proximity
from file_handler.file_writer import FileWriter
from file_handler.file_reader import FileReader
from preprocess.TF_IDF import create_tf_idf


def multiple(*func_list):
    '''run multiple functions as one'''
    # I can't decide if this is ugly or pretty
    return lambda *args, **kw: [func(*args, **kw) for func in func_list];
    None


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


def is_row_english(row):
    import string
    return row[0].lower() in string.ascii_lowercase


def wide_table(my_list, number_of_rows):
    h = len(my_list) * number_of_rows
    w = (len(my_list[0]) + number_of_rows - 1) // number_of_rows
    res = [["" for _ in range(w)] for _ in range(h)]
    for i in range(len(my_list)):
        for j in range(len(my_list[0])):
            res[(j % number_of_rows) * len(my_list) + i][j // number_of_rows] = my_list[i][j]
    return res


def add_table(win, my_list):
    for i in range(len(my_list)):
        win.grid_columnconfigure(i, weight=1)
    for i in range(len(my_list[0])):
        win.grid_rowconfigure(i, weight=1)
    for i in range(len(my_list[0])):
        for j in range(len(my_list)):
            e = Entry(win)
            e.grid(row=i, column=j, padx=0, pady=0, ipadx=0, ipady=0, sticky=W + E + N + S)
            e.insert(END, my_list[j][i])
            e.config(state="readonly")


def configure_size_window(win):
    win.title("MIR Project")
    win.geometry('840x500')
    win.grid_columnconfigure(1, weight=1)
    win.grid_columnconfigure(2, weight=1)
    win.grid_columnconfigure(3, weight=1)
    win.grid_rowconfigure(0, weight=1)
    win.grid_rowconfigure(1, weight=1)
    win.grid_rowconfigure(2, weight=1)
    win.grid_rowconfigure(3, weight=1)
    win.grid_rowconfigure(4, weight=1)
    win.grid_rowconfigure(5, weight=1)
    win.grid_rowconfigure(6, weight=1)
    win.grid_rowconfigure(7, weight=1)
    win.grid_rowconfigure(8, weight=1)
    win.grid_rowconfigure(9, weight=1)


stop_words = []


def configure_prepare_section(win):
    def prepare_CSV_clicked():
        global stop_words
        print("Loading...")
        filename = filedialog.askopenfilename()
        df = pd.read_csv(filename)
        df = df[['description', 'title']]
        df, stop_words = eng.prepare_text(df)
        index.add_multiple_documents(df)

        stopwords_window = Toplevel(win)
        stopwords_window.title("Stopwords found (TOP {}%)".format(eng.stop_word_ratio * 100))
        stopwords_window.geometry("800x800")
        add_table(stopwords_window, wide_table(stop_words, 6))

        parsed_document_window = Toplevel(win)
        parsed_document_window.title("Parsed Document")
        parsed_document_window.geometry("800x800")
        scrollbar = Scrollbar(parsed_document_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox1 = Listbox(parsed_document_window, yscrollcommand=scrollbar.set)
        listbox2 = Listbox(parsed_document_window, yscrollcommand=scrollbar.set)
        listbox3 = Listbox(parsed_document_window, yscrollcommand=scrollbar.set)

        listbox1.insert(END, 'Index')
        listbox2.insert(END, 'Title')
        listbox3.insert(END, 'Description')

        for i, row in df.iterrows():
            listbox1.insert(END, i + 1)
            listbox2.insert(END, ', '.join(row['title']))
            listbox3.insert(END, ', '.join(row['description']))

        listbox1.pack(side=LEFT, fill=BOTH, expand=TRUE)
        listbox2.pack(side=LEFT, fill=BOTH, expand=TRUE)
        listbox3.pack(side=LEFT, fill=BOTH, expand=TRUE)
        scrollbar.config(command=multiple(listbox1.yview, listbox2.yview, listbox3.yview))

        parsed_document_window.mainloop()
        stopwords_window.mainloop()

    btn_CSV = Button(win, text="Prepare CSV documents", command=prepare_CSV_clicked)
    btn_CSV.grid(column=1, row=0, sticky=W + E + N + S, columnspan=2)

    def prepare_XML_clicked():
        global stop_words
        print("Loading...")
        filename = filedialog.askopenfilename()
        df = XML_to_dataframe(filename)
        df = df[['description', 'title']]
        df, stop_words = per.prepare_text(df)
        index.add_multiple_documents(df)

        stopwords_window = Toplevel(win)
        stopwords_window.title("Stopwords found (TOP {}%)".format(per.stop_word_ratio * 100))
        stopwords_window.geometry("800x1200")
        add_table(stopwords_window, wide_table(stop_words, 6))

        parsed_document_window = Toplevel(win)
        parsed_document_window.title("Parsed Document")
        parsed_document_window.geometry("800x800")
        scrollbar = Scrollbar(parsed_document_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox1 = Listbox(parsed_document_window, yscrollcommand=scrollbar.set)
        listbox2 = Listbox(parsed_document_window, yscrollcommand=scrollbar.set)
        listbox3 = Listbox(parsed_document_window, yscrollcommand=scrollbar.set)

        listbox1.insert(END, 'Index')
        listbox2.insert(END, 'Title')
        listbox3.insert(END, 'Description')

        for i, row in df.iterrows():
            listbox1.insert(END, i + 1)
            listbox2.insert(END, ', '.join(row['title']))
            listbox3.insert(END, ', '.join(row['description']))

        listbox1.pack(side=LEFT, fill=BOTH, expand=TRUE)
        listbox2.pack(side=LEFT, fill=BOTH, expand=TRUE)
        listbox3.pack(side=LEFT, fill=BOTH, expand=TRUE)
        scrollbar.config(command=multiple(listbox1.yview, listbox2.yview, listbox3.yview))

        parsed_document_window.mainloop()
        stopwords_window.mainloop()

    btn_XML = Button(win, text="Prepare XML documents", command=prepare_XML_clicked)
    btn_XML.grid(column=3, row=0, sticky=W + E + N + S, columnspan=1)


def configure_change_index_section(win):
    entry_add_doc_desc = EntryWithPlaceholder(win, "Enter document desc, Hello! etc.")
    entry_add_doc_desc.grid(column=1, row=1, sticky=W + E + N + S, columnspan=1)
    entry_add_doc_title = EntryWithPlaceholder(win, "Enter document title, Intro etc.")
    entry_add_doc_title.grid(column=2, row=1, sticky=W + E + N + S, columnspan=1)

    def add_document_clicked():
        desc = entry_add_doc_desc.get()
        title = entry_add_doc_title.get()
        if is_row_english(desc):
            lang = eng
        else:
            lang = per
        id = index.add_single_document(stopwords_core.remove_stop_words(lang.clean_raw(desc), stop_words),
                                       stopwords_core.remove_stop_words(lang.clean_raw(title), stop_words))
        tk.messagebox.showinfo(title="Info", message="Your enter document added with ID {}".format(id))

    btn_add_doc = Button(win, text="Add single document", command=add_document_clicked)
    btn_add_doc.grid(column=3, row=1, sticky=W + E + N + S, columnspan=1)

    entry_delete_doc = EntryWithPlaceholder(win, "Enter document ID, 32198 etc.")
    entry_delete_doc.grid(column=1, row=2, sticky=W + E + N + S, columnspan=2)

    def remove_document_clicked():
        document = int(entry_delete_doc.get())
        if len(index.doc_is_available) <= document or not index.doc_is_available[document]:
            tk.messagebox.showerror(title="Error", message="Document you entered doesn't exists".format(document))
            return
        index.remove_document(document)
        tk.messagebox.showinfo(title="Info", message="Document with ID {} removed successfully".format(document))

    btn_delete_doc = Button(win, text="Delete single document", command=remove_document_clicked)
    btn_delete_doc.grid(column=3, row=2, sticky=W + E + N + S)


def configure_index_section(win):
    configure_change_index_section(win)
    entry_posting_list = EntryWithPlaceholder(win, "Enter term, Hello etc.")
    entry_posting_list.grid(column=1, row=3, sticky=W + E + N + S, columnspan=2)

    def show_posting_list_clicked():
        token = entry_posting_list.get()
        posting_list = []
        if index.token_exists(token):
            posting_list = index.positional[index.get_token_id(token)]
        posting_list_window = Toplevel(win)
        posting_list_window.title("Posting list of  '{}'".format(token))
        posting_list_window.geometry("450x450")
        scrollbar = Scrollbar(posting_list_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox = Listbox(posting_list_window, yscrollcommand=scrollbar.set)
        for doc_pos in posting_list:
            if not index.doc_is_available[doc_pos[0]]:
                continue
            listbox.insert(END, "Document {}:".format(doc_pos[0]))
            for position in doc_pos[1]:
                if position % 2 == 0:
                    listbox.insert(END, "Position {} of title".format(position // 2))
                else:
                    listbox.insert(END, "Position {} of description".format(position // 2))
        listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        posting_list_window.mainloop()

    btn_show_posting_list = Button(win, text="Show posting-list of a term", command=show_posting_list_clicked)
    btn_show_posting_list.grid(column=3, row=3, sticky=W + E + N + S)

    entry_term_pos = EntryWithPlaceholder(win, "Enter term, Hello etc.")
    entry_term_pos.grid(column=1, row=4, sticky=W + E + N + S, columnspan=1)
    entry_doc_pos = EntryWithPlaceholder(win, "Enter document ID, 32198 etc.")
    entry_doc_pos.grid(column=2, row=4, sticky=W + E + N + S, columnspan=1)

    def show_pos_term_clicked():
        token = entry_term_pos.get()
        document = entry_doc_pos.get()
        positions = []
        if index.token_exists(token):
            for doc_pos in index.positional[index.get_token_id(token)]:
                if str(doc_pos[0]) == document and index.doc_is_available[doc_pos[0]]:
                    positions = doc_pos[1]
                    break
        position_list_window = Toplevel(win)
        position_list_window.title("Positions '{}' appeared in document '{}'".format(token, document))
        position_list_window.geometry("450x400")
        scrollbar = Scrollbar(position_list_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox = Listbox(position_list_window, yscrollcommand=scrollbar.set)
        for position in positions:
            if position % 2 == 0:
                listbox.insert(END, "Position {} of title".format(position // 2))
            else:
                listbox.insert(END, "Position {} of description".format(position // 2))
        listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        position_list_window.mainloop()

    btn_show_pos_term_doc = Button(win, text="Show positions of term in document", command=show_pos_term_clicked)
    btn_show_pos_term_doc.grid(column=3, row=4, sticky=W + E + N + S)

    entry_bigram_terms = EntryWithPlaceholder(win, "Enter bigram terms, ba etc.")
    entry_bigram_terms.grid(column=1, row=5, sticky=W + E + N + S, columnspan=2)

    def show_bigram_list_clicked():
        biword = entry_bigram_terms.get()
        tokens = []
        if biword in index.bigram:
            for token_id in index.bigram[biword]:
                tokens.append(index.all_tokens[token_id])
        token_list_window = Toplevel(win)
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

    btn_show_term_bigram = Button(win, text="Show terms fit in this bigram", command=show_bigram_list_clicked)
    btn_show_term_bigram.grid(column=3, row=5, sticky=W + E + N + S)


def configure_correct_query_section(win, entry_query):
    def correct_query_clicked():
        query = entry_query.get()
        if is_row_english(query):
            lang = eng
        else:
            lang = per
        cleaned_query, corrected_query, jaccard_distance, edit_distance = spell_checker.correct(query, index,
                                                                                                stop_words[0],
                                                                                                lang)

        corrected_query_window = Toplevel(win)
        corrected_query_window.title("Corrected query")
        corrected_query_window.geometry("350x100")

        listbox = Listbox(corrected_query_window)
        listbox.insert(END, "Original query: {}".format(query))
        listbox.insert(END, "Cleaned query: {}".format(cleaned_query))
        listbox.insert(END, "Corrected query: {}".format(corrected_query))
        listbox.insert(END, "Jaccard distance: {}".format(jaccard_distance))
        listbox.insert(END, "Edit distance: {}".format(edit_distance))
        listbox.pack(side=LEFT, fill=BOTH, expand=True)
        entry_query.delete(0, "end")
        entry_query.insert(0, corrected_query)

    btn_correct = Button(win, text="Correct my query", command=correct_query_clicked)
    btn_correct.grid(column=1, row=8, sticky=W + E + N + S, columnspan=1)


def configure_search_section(win, entry_query):
    def show_score_table(score_documents, title_of_window):
        search_results_window = Toplevel(win)
        search_results_window.title(title_of_window)
        search_results_window.geometry("400x500")

        listbox = Listbox(search_results_window)
        for document, score in score_documents:
            listbox.insert(END, "Document ID: {} with similarity {}".format(document, score))
        listbox.pack(side=LEFT, fill=BOTH, expand=True)

    def lncltc_search_clicked():
        query = entry_query.get()
        if is_row_english(query):
            lang = eng
        else:
            lang = per
        tokens = stopwords_core.remove_stop_words(lang.tokenize_raw(query), stop_words)
        token_ids = [index.get_token_id(token) for token in tokens]
        score_documents = LNC_LTC.search(token_ids, index)
        show_score_table(score_documents, "Search results")

    btn_search_lnc = Button(win, text="LNC-LTC search", command=lncltc_search_clicked)
    btn_search_lnc.grid(column=2, row=8, sticky=W + E + N + S, columnspan=1)

    entry_window_size = EntryWithPlaceholder(win, "Enter your window size, 5 etc.")
    entry_window_size.grid(column=3, row=7, sticky=W + E + N + S, columnspan=1)

    def proximity_search_clicked():
        query = entry_query.get()
        window_size = int(entry_window_size.get())
        if is_row_english(query):
            lang = eng
        else:
            lang = per
        tokens = stopwords_core.remove_stop_words(lang.tokenize_raw(query), stop_words)
        token_ids = [index.get_token_id(token) for token in tokens]
        score_title_docs, score_desc_docs = proximity.search(token_ids, index, window_size)
        show_score_table(score_title_docs, "Search results on titles")
        show_score_table(score_desc_docs, "Search results on descriptions")

    btn_search_prox = Button(win, text="Proximity search", command=proximity_search_clicked)
    btn_search_prox.grid(column=3, row=8, sticky=W + E + N + S, columnspan=1)


def configure_save_load_section(win):
    OPTIONS = [
        "Without Compression",
        "Variable-byte",
        "Gamma-code"
    ]
    variable = StringVar(win)
    variable.set(OPTIONS[0])  # default value
    dropdown_menu = OptionMenu(win, variable, *OPTIONS)
    dropdown_menu.grid(column=1, row=6, sticky=W + E + N + S, columnspan=1)

    def save():
        compress_type = ""
        if variable.get() == OPTIONS[0]:
            compress_type = "none"
        if variable.get() == OPTIONS[1]:
            compress_type = "variable_byte"
        if variable.get() == OPTIONS[2]:
            compress_type = "gamma_code"
        file_writer = FileWriter(stop_words, index.doc_is_available, index.normalize_doc, index.all_tokens,
                                 index.bigram,
                                 index.positional)
        file_writer.write(compress_type)
        size_window = Toplevel(win)
        size_window.title("{}".format(variable.get()))
        size_window.geometry("400x100")
        size_label_1 = Label(size_window, text="Bigram:\nbefore = {}B, after = {}B".format(
            *file_writer.get_bigram_size()))
        size_label_2 = Label(size_window, text="Positional:\nbefore = {}B, after = {}B".format(
            *file_writer.get_positional_size()))
        size_label_1.pack()
        size_label_2.pack()
        size_window.mainloop()

    def load():
        global stop_words
        compress_type = ""
        if variable.get() == OPTIONS[0]:
            compress_type = "none"
        if variable.get() == OPTIONS[1]:
            compress_type = "variable_byte"
        if variable.get() == OPTIONS[2]:
            compress_type = "gamma_code"
        file_reader = FileReader()
        file_reader.read(compress_type)
        index.doc_is_available = file_reader.doc_is_available
        index.normalize_doc = file_reader.normalized_docs
        index.all_tokens = file_reader.all_tokens
        index.bigram = file_reader.bigram
        index.positional = file_reader.positional
        index.token_map = file_reader.token_map
        stop_words = file_reader.stop_words
        tk.messagebox.showinfo(title="Done", message="Load successfully")

    btn_save = Button(win, text="Save index", command=save)
    btn_save.grid(column=2, row=6, sticky=W + E + N + S, columnspan=1)
    btn_load = Button(win, text="Load index", command=load)
    btn_load.grid(column=3, row=6, sticky=W + E + N + S, columnspan=1)


def configure_classification_section(win):
    def train_models():
        filename = filedialog.askopenfilename()
        df = pd.read_csv(filename)
        df = df[['description', 'title']]
        df, st_wds = eng.prepare_text(df)
        df['text'] = df['description'] + df['title']
        del df['description']
        del df['title']
        tf_idf = create_tf_idf(df)
        print(tf_idf)
        pass

    btn_train = Button(win, text="Train Models", command=train_models)
    btn_train.grid(column=1, row=9, sticky=W + E + N + S, columnspan=1)


def initial_window(win):
    configure_size_window(win)
    configure_prepare_section(win)
    configure_index_section(win)
    configure_save_load_section(win)

    entry_query = EntryWithPlaceholder(win, "Enter your query, Shakespeare book etc.")
    entry_query.grid(column=1, row=7, sticky=W + E + N + S, columnspan=2)

    configure_correct_query_section(win, entry_query)
    configure_search_section(win, entry_query)
    configure_classification_section(win)


window = tkinter.Tk()
initial_window(window)
window.mainloop()
