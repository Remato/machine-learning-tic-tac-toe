### ----------------- CLASSIFICADOR ----------------- ###
import numpy as np
import pandas as pd 
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def preprocess_inputs(df):
    df = df.copy()
    
    # codificando os valores negativo / positivo como {0, 1}
    df['V10'] = df['V10'].replace({'negative': 0, 'positive': 1})
    
    # codificando todas as outras colunas como Vn, tal que n = i;
    df = onehot_encode(
        df,
        columns=['V' + str(i) for i in range(1, 10)]
    )
    
    # Split df em x -> y
    y = df['V10'].copy()
    X = df.drop('V10', axis=1).copy()
    
    # Treinando para retornar os valores
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
    return X_train, X_test, y_train, y_test

def onehot_encode(df, columns):
    df = df.copy()
    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


### ----------------- Inicialização ----------------- ###
data = pd.read_csv('./data/tic-tac-toe.csv')
X_train, X_test, y_train, y_test = preprocess_inputs(data)

try:
    with open('./data/trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
except:
    model = GaussianNB()

    # Treinando o modelo .fit()
    model.fit(X_train, y_train)
    print("trained done.\n")

    # Adicionando persistência ao modelo (salvando)
    pkl_file = "./data/trained_model.pkl"
    with open(pkl_file, 'wb') as file:
        pickle.dump(model, file)

# print('Naive Bayes' + " Accuracy: {:.2f}%".format(model.score(X_test, y_test) * 100))

### ----------------- JOGO ----------------- ###
from tkinter import Tk, StringVar, PhotoImage, Button, Label, OUTSIDE, LabelFrame
import tkinter.messagebox

root = Tk()
root.title("Jogo da Velha")
root.configure(background= '#1e3743')
root.geometry("600x300")
root.resizable(False, False)

click = True
count = 0

# DataFrame inicial do jogo, tudo vazio.
# cada posição da matriz possui 3 estados {b,o,x}
# b = blank
game = {'V1_b': [1], 'V1_o': [0], 'V1_x': [0], #primeiro elemento linha 1
        'V2_b': [1], 'V2_o': [0], 'V2_x': [0], #segundo elemento linha 1
        'V3_b': [1], 'V3_o': [0], 'V3_x': [0], #terceiro elemento linha 1

        'V4_b': [1], 'V4_o': [0], 'V4_x': [0], #primeiro elemento linha 2
        'V5_b': [1], 'V5_o': [0], 'V5_x': [0], #segundo elemento linha 2
        'V6_b': [1], 'V6_o': [0], 'V6_x': [0], #terceiro elemento linha 2

        'V7_b': [1], 'V7_o': [0], 'V7_x': [0], #primeiro elemento linha 3
        'V8_b': [1], 'V8_o': [0], 'V8_x': [0], #segundo elemento linha 3
        'V9_b': [1], 'V9_o': [0], 'V9_x': [0]} #terceiro elemento linha 3

# String dinamincas para valores 
btn1 = StringVar()
btn2 = StringVar()
btn3 = StringVar()
btn4 = StringVar()
btn5 = StringVar()
btn6 = StringVar()
btn7 = StringVar()
btn8 = StringVar()
btn9 = StringVar()

play1 = StringVar()
play2 = StringVar()

xPhoto = PhotoImage(file = './assets/X.png')
oPhoto = PhotoImage(file = './assets/O.png')

ProbabilidadePlay1 = LabelFrame(root, text="Jogador X", borderwidth=1,relief="solid")
ProbabilidadePlay1.place(x=300, y=100, width=100, height =50)
label_ProbabilidadePlay1= Label(ProbabilidadePlay1, textvariable=play1)
label_ProbabilidadePlay1.pack()

ProbabilidadePlay2 = LabelFrame(root, text="Jogador O", borderwidth=1,relief="solid")
ProbabilidadePlay2.place(x=300, y=200, width=100, height=50)
label_ProbabilidadePlay2= Label(ProbabilidadePlay2, textvariable=play2)
label_ProbabilidadePlay2.pack()

def matrix():
                
    button1 = Button(root,height=4,width=10,bd=.5,relief = 'ridge',bg = '#363636',textvariable = btn1, command=lambda: press(1,100,100)) 
    button1.place(bordermode= OUTSIDE,height=50, width=50,x=100, y=100)

    button2 = Button(root,height=4,width=10,bd = .5,relief = 'ridge',bg = '#363636',textvariable = btn2, command=lambda: press(2,150,100))
    button2.place(bordermode= OUTSIDE,height=50, width=50,x=150, y=100)
        
    button3 = Button(root,height=4,width=10,bd = .5,relief = 'ridge',bg = '#363636',textvariable = btn3, command=lambda: press(3,200,100))
    button3.place(bordermode= OUTSIDE,height=50, width=50,x=200, y=100)
        
    button4 = Button(root,height=4,width=10,bd = .5,relief = 'ridge',bg = '#363636',textvariable = btn4,command=lambda: press(4,100,150))
    button4.place(bordermode= OUTSIDE,height=50, width=50,x=100, y=150)

    button5 = Button(root,height=4,width=10,bd = .5,relief = 'ridge',bg = '#363636',textvariable = btn5,command=lambda: press(5,150,150))
    button5.place(bordermode= OUTSIDE,height=50, width=50,x=150, y=150)

    button6 = Button(root,height=4,width=10,bd = .5,relief = 'ridge',bg = '#363636',textvariable = btn6,command=lambda: press(6,200,150))
    button6.place(bordermode= OUTSIDE,height=50, width=50,x=200, y=150)

    button7 = Button(root,height=4,width=10,bd = .5,relief = 'ridge',bg = '#363636',textvariable = btn7,command=lambda: press(7,100,200))
    button7.place(bordermode= OUTSIDE,height=50, width=50,x=100, y=200)

    button8 = Button(root,height=4,width=10,bd = .5,relief = 'ridge',bg = '#363636',textvariable = btn8,command=lambda: press(8,150,200))
    button8.place(bordermode= OUTSIDE,height=50, width=50,x=150, y=200)

    button9 = Button(root,height=4,width=10,bd = .5,relief = 'ridge',bg = '#363636',textvariable = btn9,command=lambda: press(9,200,200))
    button9.place(bordermode= OUTSIDE,height=50, width=50,x=200, y=200)

def press(num,r,c):
    global click,count

    if click == True:
        labelPhoto = Label(root,image = xPhoto)
        labelPhoto.place(bordermode= OUTSIDE, height=50, width=50, x=r, y=c)
        if num == 1:
            btn1.set('X')
            game['V1_b'] = [0]
            game['V1_x'] = [1]
        elif num == 2:
            btn2.set('X')
            game['V2_b'] = [0]
            game['V2_x'] = [1]          
        elif num == 3:
            btn3.set('X') 
            game['V3_b'] = [0]
            game['V3_x'] = [1]          
        elif num == 4:
            btn4.set('X')
            game['V4_b'] = [0]
            game['V4_x'] = [1]
        elif num == 5:
            btn5.set('X')
            game['V5_b'] = [0]
            game['V5_x'] = [1]
        elif num == 6:
            btn6.set('X')
            game['V6_b'] = [0]
            game['V6_x'] = [1]
        elif num == 7:
            btn7.set('X')
            game['V7_b'] = [0]
            game['V7_x'] = [1]
        elif num == 8:
            btn8.set('X')
            game['V8_b'] = [0]
            game['V8_x'] = [1]
        else:
            btn9.set('X')
            game['V9_b'] = [0]
            game['V9_x'] = [1]
        count += 1
        click = False            
        checkWin()
            
    else:
        labelPhoto = Label(root,image = oPhoto)
        labelPhoto.place(bordermode= OUTSIDE,height=50, width=50,x= r, y= c)
        if num == 1:
            btn1.set('O')
            game['V1_b'] = [0]
            game['V1_o'] = [1]
        elif num == 2:
            btn2.set('O')
            game['V2_b'] = [0]
            game['V2_o'] = [1]
        elif num == 3:
            btn3.set('O')
            game['V3_b'] = [0]
            game['V3_o'] = [1]
        elif num == 4:
            btn4.set('O')
            game['V4_b'] = [0]
            game['V4_o'] = [1]
        elif num == 5:
            btn5.set('O')
            game['V5_b'] = [0]
            game['V5_o'] = [1]
        elif num == 6:
            btn6.set('O')
            game['V6_b'] = [0]
            game['V6_o'] = [1]
        elif num == 7:
            btn7.set('O')
            game['V7_b'] = [0]
            game['V7_o'] = [1]
        elif num == 8:
            btn8.set('O')
            game['V8_b'] = [0]
            game['V8_o'] = [1]
        else:
            btn9.set('O')
            game['V9_b'] = [0]
            game['V9_o'] = [1]
        count += 1
        click = True
        checkWin()

def checkWin():
        
    global count,click

    if (btn1.get() == 'X' and btn2.get() == 'X' and btn3.get() == 'X' or
        btn4.get() == 'X' and btn5.get() == 'X' and btn6.get() == 'X' or
        btn7.get() == 'X' and btn8.get() == 'X' and btn9.get() == 'X' or
        btn1.get() == 'X' and btn4.get() == 'X' and btn7.get() == 'X' or
        btn2.get() == 'X' and btn5.get() == 'X' and btn8.get() == 'X' or
        btn3.get() == 'X' and btn6.get() == 'X' and btn9.get() == 'X' or
        btn1.get() == 'X' and btn5.get() == 'X' and btn9.get() == 'X' or
        btn3.get() == 'X' and btn5.get() == 'X' and btn7.get() == 'X'):
        tkinter.messagebox.showinfo("Jogo da Velha", 'X Venceu!')
        click = True
        count = 0
        clear()
        matrix()
            
    elif (btn1.get() == 'O' and btn2.get() == 'O' and btn3.get() == 'O' or
        btn4.get() == 'O' and btn5.get() == 'O' and btn6.get() == 'O' or
        btn7.get() == 'O' and btn8.get() == 'O' and btn9.get() == 'O' or
        btn1.get() == 'O' and btn4.get() == 'O' and btn7.get() == 'O' or
        btn2.get() == 'O' and btn5.get() == 'O' and btn8.get() == 'O' or
        btn3.get() == 'O' and btn6.get() == 'O' and btn9.get() == 'O' or
        btn1.get() == 'O' and btn5.get() == 'O' and btn9.get() == 'O' or
        btn3.get() == 'O' and btn5.get() == 'O' and btn7.get() == 'O'):
        tkinter.messagebox.showinfo("Jogo da Velha", 'O Venceu!')
        count = 0
        clear()
        matrix()
            
    elif (count == 9):
        tkinter.messagebox.showinfo("Jogo da velha", 'Empate!')
        click = True
        count = 0
        clear()
        matrix()

    # Prevendo resultado apartir do valor atual do tabuleiro
    
    df = pd.DataFrame(data=game)
    prob = model.predict_proba(df)
    play1.set(str(round(prob[0][1]*100, 4)) + '%')
    play2.set(str(round(prob[0][0]*100, 4)) + '%')

def clear():
    btn1.set('')
    btn2.set('')
    btn3.set('')
    btn4.set('')
    btn5.set('')
    btn6.set('')
    btn7.set('')
    btn8.set('')
    btn9.set('')

    # limpando o datafram do jogo
    for i in range(9):
        game['V'+str(i+1)+'_b'] = 1
        game['V'+str(i+1)+'_o'] = 0
        game['V'+str(i+1)+'_x'] = 0

    play1.set('')
    play2.set('')


matrix()
root.mainloop()
