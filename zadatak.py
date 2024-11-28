import pandas as pd
from sklearn.model_selection import train_test_split
from utils_nans1 import *

#--------------------------------------------------------------
#Ucitavamo podatke u "df" i brisemo elemente sa praznim poljima
df = pd.read_csv('data/train.csv', sep = ',')
df = df.dropna()

#Fitujemo model
x = df.drop(columns=['plata'])
y = df['plata']

model = get_fitted_model(x, y)

#Poredimo sa test skupom
df_test = pd.read_csv('data/test.csv', sep = ',')

x_test = df_test.drop(columns=['plata'])
y_test = df_test['plata']

test_rmse = get_rmse(model, x_test, y_test)
print(test_rmse)
#--------------------------------------------------------------

#--------------------------------------------------------------
#Trazimo min i max
min, max = get_conf_interval(model, 'zvanje', alpha = 0.05)
print(min)
print(max)

#Provera validnosti
autocorrelation, _ = independence_of_errors_assumption(model, sm.add_constant(x), y, plot = False)
if autocorrelation is None:
    print("Vrednosti su validne - zadovoljena pretpostavka o nezavisnosti")
else:
    print("Vrednosti nisu validne")
#--------------------------------------------------------------

#--------------------------------------------------------------
#Ucitavamo sve podatke i vrsimo interpolaciju umesto brisanja vrednosti (menjamo metode da se ispunjavaju uslovi)
df = pd.read_csv('data/train.csv', sep = ',')
df['zvanje'] = df['zvanje'].interpolate(method='spline', order=3, limit_direction='both')
df['godina_doktor'] = df['godina_doktor'].interpolate(method='linear', limit_direction='both')

#Brisanje nepotrebnih kolona (uskladjujemo sa uslovima)
df = df.drop(columns=['pol Muski', 'pol Zenski'])

#Delimo ih u odnos 80/20
x = df.drop(columns=['plata'])
y = df['plata']
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=42)

#Fitujemo model traina
model = get_fitted_model(x_train, y_train)
#print(model.summary())

#Provera da li su zadovoljene pretpostavke
print(are_assumptions_satisfied(model, x_train, y_train))

#Mera na validacionom skupu
print(get_rmse(model, x_val, y_val))

#Mera na test skupu
df_test = pd.read_csv('data/test.csv', sep=',')
x_test = df_test.drop(columns=['plata', 'pol Muski', 'pol Zenski'])
y_test = df_test['plata']

print(get_rmse(model, x_test, y_test))
#--------------------------------------------------------------