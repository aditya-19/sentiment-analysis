# -*- coding: utf-8 -*-
"""
Created on Tue May 26 02:29:54 2020

@author: Aditya
"""
import tkinter as tk
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *

root=tk.Tk()
root.title("Sentiment Analysis of Tweets")
root.geometry("500x300+350+250")

lblEnterKeyword = Label(root,text="Enter your Search keyword.",font=("Verdana",18))
lblEnterKeyword.pack(pady=20)
entEnterKeyword = Entry(root , bd=5, width=50)
entEnterKeyword.pack(pady=20,ipady=5)
keyword=entEnterKeyword.get()

#==========new===========#
new = Toplevel(root)
new.title("Sentiment Analysis")
new.geometry("1200x700+350+250")
new.withdraw()
#==========new===========#
def plot():
    pos=47
    neu=1
    neg=52
    data1={'Labels':['Positive Reviews','Neutral Reviews','Negative Reviews'],
           'Values' :[pos,neu,neg]
           }
    data1_df=pd.DataFrame(data1,columns=['Labels','Values'])
    figure1 = plt.Figure(figsize=(6,7))
    ax1 = figure1.add_subplot(111)
    ax1.bar(data1_df['Labels'], data1_df['Values'])
    ax1.set_title('Sentiment Distribution of Tweets')
    bar1 = FigureCanvasTkAgg(figure1, new)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    
    figure2 = plt.Figure(figsize=(13,18))
    explode = (0, 0, 0) #To separate the slices in pir-chart
    a = figure2.add_subplot(111)
    a.pie(data1_df['Values'], explode=explode, labels=data1_df['Labels'], autopct='%1.1f%%',
    shadow=True, startangle=90)
    a.axis('equal')
    a.set_title('Sentiment Distribution of Tweets')
    pie1 = FigureCanvasTkAgg(figure2, new)
    pie1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)


def backto_root():
	new.withdraw()
	root.deiconify()
btnBack = Button(new, text="Back", width=20,font=("Verdana",15),command=backto_root)
btnBack.pack(pady=20)

def goto_new():
    root.withdraw()
    new.deiconify()
    plot()
btnSearch = Button(root, text="Show Sentiments", width=20,command=goto_new,font=("Verdana",15))
btnSearch.pack(pady=20)


root.mainloop()

#labels = 'Positive Review', 'Negative Review', 'Neutral Review'
#sizes = [pos,neg,neu]
#explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

#fig1, ax1 = plt.subplots()
#ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
 #   shadow=True, startangle=90)
#ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#plt.show()

#labels_list=["Positive Review","Negative Review","Neutral Review"]
#index = np.arange(len(labels_list))
#plt.bar(index, sizes)
#plt.xlabel('Sentiments', fontsize=15)
#plt.ylabel('No of Reviews', fontsize=15)
#plt.xticks(index, labels_list, fontsize=10, rotation=0)
#plt.title('Sentiment Analysis Distribution of Tweets')
#plt.show()