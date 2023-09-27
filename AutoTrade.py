import json
import time
from datetime import  datetime
import urllib 
import requests
import pandas as pd
import numpy as np
import hashlib
import hmac
from  tensorflow import keras
from tensorflow import math


#works for spot market

apikey = "your key"
secret = "your secret key"
test = requests.get("https://api.binance.com/api/v3/ping")
servertime = requests.get("https://api.binance.com/api/v3/time")

servertimeobject = json.loads(servertime.text)
servertimeint = servertimeobject['serverTime']

tick=0.00001

#buy base coin by market order
def openO(money,base):
    params={'symbol':base+'USDT',
            'side':'BUY',
            'type':'MARKET',
            'quoteOrderQty':money,
            'newOrderRespType':'RESULT',
            'timestamp':int(time.mktime(datetime.now().timetuple())*1000)}
    message=urllib.parse.urlencode(params)
    signature=hmac.new(secret.encode('utf-8'),message.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature']=signature
    head={"X-MBX-APIKEY" : apikey}
    OrderOpen = requests.post("https://api.binance.com/api/v3/order",     params = params, headers = head)
    print(OrderOpen,OrderOpen.text)
    qty=float(json.loads(OrderOpen.text)['executedQty'])
    return qty

#sell base coin by market order
def closeO(CRYPTO,base):
    params={'symbol':base+'USDT',
            'side':'SELL',
            'type':'MARKET',
            'quantity':CRYPTO,
            'newOrderRespType':'RESULT',
            'timestamp':int(time.mktime(datetime.now().timetuple())*1000)}
    message=urllib.parse.urlencode(params)
    signature=hmac.new(secret.encode('utf-8'),message.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature']=signature
    head={"X-MBX-APIKEY" : apikey}
    OrderClose= requests.post("https://api.binance.com/api/v3/order",     params = params, headers = head)
    print(OrderClose,OrderClose.text)
    if OrderClose.status_code == 400:
        print(OrderClose,OrderClose.text)
        return closeO(CRYPTO,base)
    print(OrderClose,OrderClose.text)
    qty=float(json.loads(OrderClose.text)['cummulativeQuoteQty'])    
    return qty

# cancel limit order
def cancelO(SYMBOL,orderId):
    params={'symbol':SYMBOL+'USDT',
            'orderId':orderId,
            'timestamp':int(time.mktime(datetime.now().timetuple())*1000)}
    message=urllib.parse.urlencode(params)
    signature=hmac.new(secret.encode('utf-8'),message.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature']=signature
    head={"X-MBX-APIKEY" : apikey}
    OrderClose= requests.delete("https://api.binance.com/api/v3/order",     params = params, headers = head)
    if OrderClose.status_code == 400:
        print(OrderClose,OrderClose.text)
        time.sleep(1)
        return False
    print(OrderClose,OrderClose.text)   
    return True

#open limit order to buy base coin
def limitOpen(SYMBOL,QUANTITY, price):
    params={'symbol':SYMBOL+'USDT',
            'side':'BUY',
            'type':'LIMIT',
            'quantity':QUANTITY,
            'price':price,
            'timeInForce':'GTC',
            'newOrderRespType':'ACK',
            'newClientOrderId':'1',
            'timestamp':int(time.mktime(datetime.now().timetuple())*1000)}
    message=urllib.parse.urlencode(params)
    signature=hmac.new(secret.encode('utf-8'),message.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature']=signature
    head={"X-MBX-APIKEY" : apikey}
    Order= requests.post("https://api.binance.com/api/v3/order",     params = params, headers = head)
    print(Order,Order.text)
    if Order.status_code==400:
        return limitOpen(QUANTITY-tick, price)    
    return json.loads(Order.text)['orderId']

#open limit order to sell base coin
def limitClose(SYMBOL,QUANTITY, price):
    params={'symbol':SYMBOL+'USDT',
            'side':'SELL',
            'type':'LIMIT',
            'quantity':QUANTITY,
            'price':price,
            'timeInForce':'GTC',
            'newOrderRespType':'ACK',
            'newClientOrderId':'1',
            'timestamp':int(time.mktime(datetime.now().timetuple())*1000)}
    message=urllib.parse.urlencode(params)
    signature=hmac.new(secret.encode('utf-8'),message.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature']=signature
    head={"X-MBX-APIKEY" : apikey}
    Order= requests.post("https://api.binance.com/api/v3/order",     params = params, headers = head)
    print(Order,Order.text)
    if Order.status_code==400:
        return limitClose(SYMBOL,QUANTITY, price)    
    return json.loads(Order.text)['orderId']


        

def checkOrder(SYMBOL,ID):
    params={'symbol':SYMBOL+'USDT',
            'orderId':ID,
            'timestamp':int(time.mktime(datetime.now().timetuple())*1000)}
    message=urllib.parse.urlencode(params)
    signature=hmac.new(secret.encode('utf-8'),message.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature']=signature
    head={"X-MBX-APIKEY" : apikey}
    OrderClose= requests.get("https://api.binance.com/api/v3/order",     params = params, headers = head)
    print(OrderClose,OrderClose.text)
    status=json.loads(OrderClose.text)['status']
    if status=='FILLED':
        return True
    return False
    
#check immediate price of base coin
def ticker(SYMBOL):
    params={'symbol':SYMBOL+'USDT'}
    try:
        gettick = requests.get("https://api.binance.com/api/v3/ticker/price",     params = params)
    except requests.exceptions.ConnectionError:
        print('Connection error, Cooling down for 1 sec...')
        time.sleep(1)
        return ticker(SYMBOL)
 
    except requests.exceptions.Timeout:
        print('Timeout, Cooling down for 5 secs...')
        time.sleep(5 )
        return ticker(SYMBOL)
 
    except requests.exceptions.ConnectionResetError:
       print('Connection reset by peer, Cooling down for 5 min...')
       time.sleep(5 * 60)
       return ticker()
    if gettick.status_code == 200:
       price=float(json.loads(gettick.text)['price'])
       return price
    print(f'Got erroneous response back: {gettick}')
    return ticker(SYMBOL)





W=256



#reproducing data processing used to train NN
def drawing(A):
    rez=np.zeros((12,W))
    HZ=np.zeros((W))
    for t in range(W):
        HZ[t]=A[t,4]*(A[t,3]-A[t,0])/(2*(A[t,1]-A[t,2])-abs(A[t,3]-A[t,0])+0.000001)
    M=np.max(np.abs(HZ))
    for t in range(W):
        HZ[t]=np.sum(HZ[t:W-10])
    m1=np.median(A[:W-10,4])
    m2=np.median(A[:W-10,6])
    for t in range(W):
        rez[7,t]=A[t,7]/A[t,4]  
    for t in range(W):
        rez[4,t]=np.log(A[t,4]/m1/2)
        rez[6,t]=np.log(A[t,6]/m2/2)        
    for t in range(W):
            rez[11,t]=10*(2*(A[t,1]-A[t,2])-abs(A[t,3]-A[t,0]))/A[W-1,3]-10
            rez[0,t]=A[t,4]/rez[11,t]
            rez[5,t]=HZ[t]/M
    for channel in (1,2,3,8,9,10):
         for t in range(W):
             rez[channel,t]=10*A[t,channel]/A[W-1,3]-10
    return rez
     








#obtain current data from server
def get_data(symbol, interval='1m', start_time=None, limit=W+99):
    API_BASE = 'https://api.binance.com/api/v3/'

    LABELS = [
        'open_time',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close_time',
        'quote_asset_volume',
        'number_of_trades',
        'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume',
        'ignore'
    ]

    params = {
        'symbol': symbol+'USDT',
        'interval': interval,
        'startTime': start_time,
        'limit': limit
    }
    try:
        response = requests.get(f'{API_BASE}klines', params, timeout=5)
    except requests.exceptions.ConnectionError:
        print('Connection error, Cooling down for 5 secs...')
        time.sleep(5 )
        return get_data(symbol, interval, start_time, limit)
    
    except requests.exceptions.Timeout:
        print('Timeout, Cooling down for 5 secs...')
        time.sleep(5)
        return get_data(symbol, interval, start_time, limit)
    
    except requests.exceptions.ConnectionResetError:
        print('Connection reset by peer, Cooling down for 5 min...')
        time.sleep(5 * 60)
        return get_data(symbol, interval, start_time, limit)

    if response.status_code == 200:
        return pd.DataFrame(response.json(), columns=LABELS).drop(columns=['open_time','close_time', 'ignore'])
    print(f'Got erroneous response back: {response}')
    return pd.DataFrame([])

#processing data from server
def process_data(lines181):
    Rez=np.zeros((W,12))
    i=99
    a=lines181.to_numpy(dtype=float)
    ma100=np.sum(a[:100,3])/100
    ma25=np.sum(a[75:100,3])/25
    vwap=a[i,5]/a[i,4]
    Rez[0,0]=a[i,0]
    Rez[0,1]=a[i,1]
    Rez[0,2]=a[i,2]
    Rez[0,3]=a[i,3]
    Rez[0,4]=a[i,4]
    Rez[0,5]=a[i,5]
    Rez[0,6]=a[i,6]
    Rez[0,7]=a[i,7]
    Rez[0,8]=ma100
    Rez[0,9]=ma25
    Rez[0,10]=vwap
    for i in range(100,W+99):
        ma100=ma100+a[i,3]/100-a[i-100,3]/100
        ma25=ma100+a[i,3]/25-a[i-25,3]/25
        vwap=a[i,5]/a[i,4]
        ii=i-99
        Rez[ii,0]=a[i,0]
        Rez[ii,1]=a[i,1]
        Rez[ii,2]=a[i,2]
        Rez[ii,3]=a[i,3]
        Rez[ii,4]=a[i,4]
        Rez[ii,5]=a[i,5]
        Rez[ii,6]=a[i,6]
        Rez[ii,7]=a[i,7]
        Rez[ii,8]=ma100
        Rez[ii,9]=ma25
        Rez[ii,10]=vwap
    RRez=drawing(Rez)
    return RRez







#incorporating custom layers to the NN
class PASS(keras.layers.Layer):
    def __init__(self, initializer, kernel_regularizer=None,**kwargs):
        super(PASS, self).__init__(**kwargs)
        self.initializer=initializer
        self.kernel_regularizer=kernel_regularizer
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1]),
            initializer=self.initializer,
            trainable=True, regularizer=self.kernel_regularizer)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initializer': self.initializer,
            'kernel_regularizer': self.kernel_regularizer,
        })
        return config   
         
    def call(self, inputs):
        return math.multiply(inputs, self.w) 

        
        
        
        
        


       
def get_pred(nn,base):
    datafromserver=get_data(base)
    BTC=get_data('BTC')
    while len(datafromserver)!=W+99:
        time.sleep(1)
        datafromserver=get_data(base)
    net_input=process_data(datafromserver)
    BTC_input=process_data(BTC)
    model=nn
    rez=model([net_input[0].reshape((1,W)),
               net_input[1].reshape((1,W)),
               net_input[2].reshape((1,W)),
               net_input[3].reshape((1,W)),
               net_input[4].reshape((1,W)),
               net_input[5].reshape((1,W)), 
               net_input[6].reshape((1,W)), 
               net_input[7].reshape((1,W)), 
               net_input[8].reshape((1,W)), 
               net_input[9].reshape((1,W)), 
               net_input[10].reshape((1,W)),
               net_input[11].reshape((1,W)) ],training=False)
    return rez.numpy()[0,0]
    
    











coinlist=['XRP','ETH']    




preds=[]

model=keras.models.load_model('ngs7.h5',custom_objects={'PASS': PASS})


USDTBAL=1000

trigger=0
counter=0



#Algorithm, that corresponds to algorithm used for simulation
while True:
        
    if trigger==0:
        print(USDTBAL)
        for coin in coinlist:
            time.sleep(1)
            rez=get_pred(model, coin)
            print('coin: ' + str(coin)+ '; NN prediction: '+str(rez))
            if rez>0.5:
                buyCoin=coin
                trigger=1
                preds.append(rez)
                break 
                
    if trigger==1: 
        BuyPrice=ticker(buyCoin)
        CRYPTOBAL=openO(USDTBAL,buyCoin)
        ID=limitClose(buyCoin,CRYPTOBAL*0.999//1, BuyPrice*1.00575)
        trigger=2
        
    if trigger==2:       
        time.sleep(60*5)
        tick=ticker(buyCoin)
        check=tick/BuyPrice
        rez=get_pred(model, coin)
        check=checkOrder(buyCoin,ID)
        if check==True:
            USDTBAL=CRYPTOBAL*0.999*BuyPrice*1.00575
            trigger=0
            continue
        if rez<0.5:
            USDTBAL=closeO(CRYPTOBAL*0.999//1,buyCoin)
            trigger=0


            
        
       
        
        
        
    










































# while True:
#     rez=get_pred()
#     ma4,niz,verh=get_ma4()
#     tp=ticker()
#     if rez>0.55:
#         timer=0
#         while True:
#             ma7,niz7,verh7,open7,close7=get_ma7()
#             tpbuy=ticker()
#             if True:
#                 BTC=openO(USDT)
#                 sdelka=1
#                 break
#             else:
#                 sdelka=0
#                 break
#         if sdelka==1:
#             counter+=1
#             while True:
#                 ma7,niz0,verh0,open70,close70=get_ma7()
#                 time.sleep(10)
#                 ma7,niz1,verh1,open71,close71=get_ma7()
#                 if (close71-open71)<0:
#                     tp=ticker()
#                     if tp>1.0021*tpbuy:
#                         closeO(BTC)
#                         break
#                 rez=get_pred()
#                 if niz0>niz1 and verh0>verh1 and rez<0.5:
#                     tp=ticker()
#                     lowend=round(tp,1)*0.9996666666
#                     highend=round((2*verh1+niz1)/3,1)
#                     if lowend>=highend:
#                         highend=lowend*1.001
#                     OCOclose(BTC,highend,lowend)
#                     while True:
#                         w=checkOrder()
#                         time.sleep(15)
#                         print(f'waiting for OCOorder to be executed {datetime.now()}')
#                         if w!=2:
#                             break
#                     while True:
#                         rez=get_pred()
#                         time.sleep(30)
#                         print(f'waiting for OCOorder to be executed {datetime.now()}')
#                         if rez<=0.1:
#                             break
#                     break
#     print(f'couner= {counter}  pred= {rez}  {datetime.now()}')
        
                
                    
