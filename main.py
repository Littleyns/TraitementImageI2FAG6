from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
import time
from PIL import Image, ImageTk
from tkinter import ttk 
from tkinter import filedialog


### init ###
global nomfichier
import os 


nomfichier = os.getcwd()+"/test.png"
img_process = ""
global conf_thres


conf_thres = 0.4
ws = Tk() 
#ws = Toplevel()
ws.title('inferrence')
ws.geometry('1000x900') 

global current_value
current_value = DoubleVar()


def open_file():
    file_path = askopenfile(mode='r', filetypes=[('Image Files', '*jpeg')])
    if file_path is not None:
        pass

def ouvrirfich():
    global nomfichier
    nomfichier=filedialog.askopenfilename()
    print(nomfichier)
    
    rgb = Image.open(nomfichier).resize((750,600))

    img2 = ImageTk.PhotoImage(rgb)
    label.configure(image=img2)
    label.image = img2
    


    
def uploadFiles():
    pb1 = Progressbar(
        ws, 
        orient=HORIZONTAL, 
        length=300, 
        mode='determinate'
        )
    pb1.place(x=350,y=830)
    from PIL import Image
    from inferrence import detect_local
    global nomfichier
    
    print(nomfichier)
    global conf_thres
    print(conf_thres)
    img_process = detect_local("../with.pt", nomfichier, 640,conf_thres, device = 'cpu')
   
    
    rgb = Image.fromarray(img_process)
    rgb = rgb.resize((750,600))
    img2 = ImageTk.PhotoImage(rgb)
    global label
    label.configure(image=img2)
    label.image = img2
    
    
    Label(ws, text='inferrence succeful', foreground='green').place(x=400,y=828)
        


def get_current_value():
    global conf_thres
    global current_value
    conf_thres = current_value.get()
    return '{: .2f}'.format(current_value.get())
    

def slider_changed(event):
    value_label.configure(text=get_current_value())





def graphic():
    adharbtn = Button(
        ws, 
        text ='choisir fichier', 

        command = ouvrirfich
        ) 

    adharbtn.place(x=430, y=80)
    im = Image.open(nomfichier).resize((750,600))
    tkim = ImageTk.PhotoImage(im)
    global label
    label = Label(ws, image = tkim)
    label.image = tkim 
    upld = Button(
        ws, 
        text='inferrence', 
        command=uploadFiles
        )
    upld.place(x=440, y=140)
    label.place(x=120, y=200)
    
    slider = ttk.Scale(
        ws,
        from_=0,
        to=1,
        orient='horizontal',  # vertical
        command=slider_changed,
        variable=current_value
    )
    slider.place(x=300,y=40,width=420)
    global value_label
    value_label = ttk.Label(
        ws,
        text=get_current_value()
    )
    value_label.place(x=200, y=35)


graphic()
ws.mainloop()
