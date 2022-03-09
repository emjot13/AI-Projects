import pandas as pd 
import matplotlib.pyplot as plt

miasta = pd.read_csv(r'inteligencja_obliczeniowa/lab1/zad3/miasta.csv')

# data = {"Rok": [2010], "Gdansk": [460], "Poznan": [555], "Szczecin": [405]}
# toAdd = pd.DataFrame(data)

# toAdd.to_csv('inteligencja_obliczeniowa/lab1/zad3/miasta.csv', mode='a', index=False, header=False)


cols_list = ["Rok", "Gdansk", "Poznan", "Szczecin"]
miasta = pd.read_csv(r'inteligencja_obliczeniowa/lab1/zad3/miasta.csv', usecols=cols_list)

plt.plot(miasta["Rok"], miasta["Gdansk"], label="Gdańsk")  
plt.plot(miasta["Rok"], miasta["Szczecin"], label="Szczecin")  
plt.plot(miasta["Rok"], miasta["Poznan"], label="Poznań")  
plt.xlabel("Lata")
plt.ylabel("Liczba ludności [tys]")
plt.legend()
plt.show()