import sys
import os
import datetime 
import pandas as pd
import numpy as np 

from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from matplotlib import style as style0
style0.use('seaborn-colorblind');

from tensorflow import keras

from sklearn.model_selection import train_test_split


class LongShortTermMemory:

    def __init__(self, df_train, df_validation, df_test, learning_rate=0.05, epochs=5000):
        self.learning_rate=learning_rate;
        self.epochs=epochs;
        self.df_train=df_train;
        self.df_validation=df_validation;
        self.df_test=df_test;
        print('epochs=',self.epochs)

    def developModels(self):
        #learning_rate0=0.03
        self.opt0=keras.optimizers.Adam(learning_rate=self.learning_rate)
        #
        list_nn=[keras.layers.RNN(keras.layers.LSTMCell(60, activation='tanh'), input_shape=[1,2], return_sequences=True), 
                 keras.layers.RNN(keras.layers.LSTMCell(60, activation='tanh'), return_sequences=True), keras.layers.Dense(1, activation='linear') ]
        self.mdl0=keras.models.Sequential(list_nn)
        #mdl0.add(keras.layers.LSTM(50, input_shape=[1,2], return_sequences=True) ) #
        #mdl0.add(keras.layers.LSTM(50, return_sequences=True) ) 
        #self.mdl0.add(keras.layers.RNN(keras.layers.LSTMCell(60, activation='tanh'), input_shape=[1,2], return_sequences=True) );
        #self.mdl0.add(keras.layers.RNN(keras.layers.LSTMCell(60, activation='tanh'), return_sequences=True) )
        #self.mdl0.add( keras.layers.Dense(1, activation='linear') )
        return -1;

    def learn(self):
        print('model=', self.mdl0.summary())
        self.mdl0.compile(loss='mse', optimizer=self.opt0, metrics=['accuracy'])
        #history0=mdl0.fit(df_train[['month0','day0']].to_numpy().reshape(-1,1,2),df_train['v1'].to_numpy().reshape(-1,1,1),epochs=1150,verbose=0)
        self.history0=self.mdl0.fit(self.df_train[['month0', 'day0']].to_numpy().reshape(-1,1,2), self.df_train['v1'].to_numpy().reshape(-1,1)
        ,epochs=self.epochs,verbose=0,batch_size=200)
        return -1;

    def forecast(self):
        #y_pred=mdl0.predict(df_test[['month0', 'day0']].to_numpy().reshape(-1,1,2))
        y_pred=self.mdl0.predict(df_test[[ 'month0','day0']].to_numpy().reshape(-1,1,2))
        y_pred=y_pred.reshape(-1);
        print('y_pred=\n', y_pred, ' \n y_pred shape=', y_pred.shape)
        self.df_test=self.df_test.reset_index(drop=True)
        print('df_test=\n', self.df_test.iloc[:5])
        self.df_test['v1_pred']=y_pred;
        return -1;

    def visualize(self):
        fig0,(ax0,ax1)=plt.subplots(figsize=(12,7), nrows=2, ncols=1)
        #pd.DataFrame(self.history0.history).plot(logy=True, ax=ax1, title='eta=' + str(self.learning_rate) )

        ax0.plot(self.df_train['date0'], self.df_train['v1']);
        ax0.plot(self.df_test['date0'], self.df_test['v1'])
        ax0.plot(self.df_test['date0'], self.df_test['v1_pred'])
        plt.tight_layout();
        plt.show();
        plt.close('all');
        return -1;

    def carryOut(self):
        #self.developModels()
        #self.learn()
        # 作成ずみのモデルを使用
        print('Now we use a pre-trained model ...')
        self.mdl0=[]
        restored_model=keras.models.load_model('Models/model_0')
        self.mdl0=restored_model
        self.forecast();
        self.visualize();
        model_stored_bool=False;
        if model_stored_bool==True:
            print('Now we save this model ...')
            # !mkdir -p saved_model
            self.mdl0.save('Models/model_1')


if __name__=='__main__':
    df_data=pd.DataFrame(pd.date_range(start='2020-01-25', end='2020-09-30', freq='D'), columns=['date0'])
    #df_data['batch0']=0;
    df_data['month0']=df_data['date0'].dt.month;
    df_data['day0']=df_data['date0'].dt.day;
    df_data['noise0']=np.random.randint(low=-4,high=4,size=(df_data.shape[0],))
    df_data['v1']= 25 * (1 + np.cos( df_data['month0'] * 0.05 ) ) +  25 * np.sin( df_data['day0'] * 0.1) / ( 2 + 1.5 * np.sin(df_data['day0'] * 0.3) ) + df_data['noise0'];
    #df_data['v1']=np.sin(df_data['day0'] * 0.2) + df_data['month0'] * 0.1
    #np.random.randint(low=-10,high=15,size=(df_data.shape[0],))
    df_train=df_data.iloc[:165,:]
    df_test=df_data.iloc[165:,:]
    print('train=\n', df_train[['date0', 'month0', 'day0', 'noise0']].head(n=5).to_numpy())
    print('shape=\n', df_train[['month0','day0']].to_numpy().shape)
    print('reshaped=\n', df_train[['month0','day0']].to_numpy().reshape(-1,1,2).shape)
    print('train=\n', df_train[['month0','day0']].head(n=5).to_numpy().reshape(-1,1,2))
    print('pwd=', os.getcwd())
    print('now=', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'));

    ins0=LongShortTermMemory(learning_rate=0.01, epochs=3000
    , df_train=df_train, df_validation=df_test, df_test=df_test)
    ins0.carryOut();


"""


def sin_(x, T0=100):
    return np.sin(2.0*np.pi * x /T0)

def toy_problem(T0=100,ampl=0.05):
    x=np.arange(0,2*T0+1)
    noise=ampl*np.random.uniform(low=-1, high=1, size=len(x))
    return sin_(x) + noise

T0=100;
f0=toy_problem(T0).astype(np.float32)
length_of_sequences=len(f0)
maxlen=25;
x0=[]
t0=[]

for i0 in range(length_of_sequences - maxlen):
    x0.append(f0[i0:i0+maxlen]);
    t0.append(f0[i0+maxlen]) 

print('B x0 shape=', x0[0:1], ' len=', len(x0))
x0=np.array(x0).reshape(-1,maxlen,1)
t0=np.array(t0).reshape(-1,1);
print('A x shape=', x0.shape)
print('A x0=', x0[0:9,0:1,0])
x_train, x_val, t_train, t_val=train_test_split(x0, t0, test_size=0.2, shuffle=False)

print('len sequences=', length_of_sequences)

mdl0=keras.models.Sequential();
mdl0.add(keras.layers.SimpleRNN(50,activation='tanh',kernel_initializer='glorot_normal',recurrent_initializer='orthogonal'))
mdl0.add(keras.layers.Dense(1, activation='linear'))

optimizer0=keras.optimizers.Adam(learning_rate=0.001)
mdl0.compile(loss='mse')
hist0=mdl0.fit(x_train, t_train, batch_size=100, verbose=2, epochs=1000)

fig0,ax0=plt.subplots(figsize=(12,7))
pd.DataFrame(hist0.history).plot(logy=True)

#ax0.plot(df_train['date0'], df_train['v1']);
#ax0.plot(df_test['date0'], df_test['v1'])
#ax0.plot(df_test['date0'], df_test['v1_pred'])
#ax0.plot(x_train, x_val)
plt.tight_layout();
plt.show();
plt.close('all');
sys.exit()
""" 

