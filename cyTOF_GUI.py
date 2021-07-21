
from tkinter import * 
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#import guidata
#import guidata.dataset.datatypes as dt
#import guidata.dataset.dataitems as di

import pandas as pd



# plot function is created for  
# plotting the graph in  
# tkinter window 
def plot(): 
        Var1=tkvar1.get()
        Var2=tkvar2.get()
        figure = Figure(figsize=(5, 4), dpi=100)
        plt = figure.add_subplot(1, 1, 1)
        plt.scatter(df[Var1], df[Var2], color='red')
        plt.set_title ("Scatter Plot)", fontsize=16)
        plt.set_ylabel(Var1, fontsize=14)
        plt.set_xlabel(Var2, fontsize=14)
        canvas = FigureCanvasTkAgg(figure, master=root)
        canvas.get_tk_widget().pack()
        canvas.draw()
        
        
        
# the main Tkinter window 
root = Tk() 
  
# setting the title  
root.title('Plotting in Tkinter') 
  
# dimensions of the main window 
root.geometry("600x600") 


mainframe = Frame(root)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)
mainframe.pack(pady = 10, padx = 10)



df=pd.read_csv("Test.csv")

Names=['Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'H3.3',
 'BMI-1',
 'c-Myc',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H4',
 'H3K27M']

tkvar1 = StringVar(root)
tkvar1.set('H3.3') # set the default option
popupMenu1 = OptionMenu(mainframe, tkvar1, *Names)
tkvar2 = StringVar(root)
tkvar2.set('H3.3') # set the default option
popupMenu2 = OptionMenu(mainframe, tkvar2, *Names)

Label(mainframe, text="Var1").grid(row = 0, column = 1)
Label(mainframe, text="Var2").grid(row = 0, column = 2)
popupMenu1.grid(row = 1, column =1)
popupMenu2.grid(row = 1, column =2)




# button that displays the plot 
plot_button = Button(master = root,  
                     command = plot, 
                     height = 2,  
                     width = 10, 
                     text = "Plot") 
  
# place the button  
# in main window 
plot_button.pack() 



# on change dropdown value
def change_dropdown(*args):
    print( tkvar1.get() )
    print( tkvar2.get() )

# link function to change dropdown
tkvar1.trace('w', change_dropdown)
tkvar2.trace('w', change_dropdown)

  
# run the gui 
root.mainloop() 
