# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:24:20 2019

@author: Sunsharp
"""
import pandas as pd
from fbprophet import Prophet
import numpy as np
import time
from sympy import *
#import sys

def predict(path):
    print('开始时间：')
    start = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    data = pd.read_excel(path)
    save_data = pd.DataFrame(columns=['areaid','date','predict_package','compute_package','percent'])
    for area_num in data.areaid.unique():
    #for area_num in [5134,]:
        df = data[(data.areaid==area_num)]      


        df['package'] = df.package.astype(float)
        df['y'] = np.log(df.package)
        df['ds'] = df.month
        #df['ds'] = pd.date_range('2015-01-01','2019-4-1',freq='M')
        
        #设置春节、电商节
        playoffs = pd.DataFrame({
                'holiday': 'playoff',
                'ds': pd.to_datetime(['2015-02-28','2016-02-29','2017-01-31','2018-02-28','2019-02-28', '2020-01-31']),
            'lower_window': 0,
            'upper_window': 1,})
            
        efestival = pd.DataFrame({
          'holiday': 'efestival',
          'ds': pd.to_datetime(['2015-11-30' ,'2016-11-30','2017-11-30', '2018-11-30','2019-11-30','2020-11-30',
                                '2015-06-30' ,'2016-06-30','2017-06-30', '2018-06-30','2019-06-30','2020-06-30',
                                '2015-12-31' ,'2016-12-31','2017-12-31', '2018-12-31','2019-12-31','2020-12-31',]),
          'lower_window': 0,
          'upper_window': 1,
        })
        
        holidays = pd.concat((playoffs,efestival))
        
        
        m = Prophet(holidays=holidays, holidays_prior_scale=10)
        m.fit(df[:-3])
        future = m.make_future_dataframe(periods=3,freq='m')
        forecast = m.predict(future)
        forecast['exp'] = np.exp(forecast.yhat)
        cache_list = forecast[forecast.ds>='2017-07-01']['exp'].tolist()
        
        #print(m.plot(forecast))
        
        #计算环比
        
        df2 = df[df.ds>='2017-07-01']
        df2['percent'] = (df2['money']-df2['money'].shift(1))/df2['money'].shift(1)
        
#        df2['package_percent'] = (df2['package2']-df2['package2'].shift(1))/df2['package2'].shift(1)

        #七月电商快递件数x1,八月x2
        x1 = Symbol('x1')
        x2 = Symbol('x2')
        #网零环比和快递业务量环比
        sell_per = df2[df2.ds=='2017-08-31']['percent'].iloc[0]#网零环比
        pack_per = df2[df2.ds=='2017-08-31']['percent'].iloc[0]#快递业务量环比
        count7 = df2[df2.ds=='2017-07-31']['package'].iloc[0]
        count8 =df2[df2.ds=='2017-08-31']['package'].iloc[0]
        
        
        #普通快递件八月环比的取值范围span
#        span = np.linspace(-2,2,200)
        if sell_per>=pack_per:
            span = np.linspace(10*(pack_per-np.abs(sell_per)),pack_per,200)
        else:
            span = np.linspace(pack_per,10*(pack_per+np.abs(sell_per)),200)
        
        agrv_n = 0.925#参数 n=1.0 在此
        diff = 0

        while diff==0:
            expend = 1
            for number in span:
                func = [x1+x2-count7,(1+sell_per*agrv_n)*x1 + (1+number)*x2-count8]
                
                name = [x1,x2]
                result = solve(func,name)
#                print('result:',result)
    
                if result!=[] and float(result[x1])>0 and float(result[x2])>0:
                    #趋近75%为最佳
                    if np.abs(float(result[x1])/float(result[x2]) - 7.5/2.5)<= np.abs(diff/(count7-diff) - 7.5/2.5):
                        diff = float(result[x1])
            span = np.linspace(span[0]-expend,span[-1]+expend,100)
            expend += 2
            if expend>10:
                print('*'*40)
                print('%s地区普通快递件八月环比无正解'%area_num)
                print('*'*40)
                break
#        检查
        if diff==0:
            diff = count7*0.72
        elif diff/count7<0.56 and len(str(area_num))==2:
            diff = count7*0.72
        elif diff/count7<0.50 and len(str(area_num))==4:
            diff = count7*0.60
        print('*'*20)
        print('diff:',diff)
        print('*'*20)
        
        compute_package = list()
        compute_package.append(diff)
        
        persent = df2.percent.tolist()
        
        np.random.seed(2)
        for t in persent[1:]:
            if t>0:
                compute_package.append((1+t*agrv_n)*compute_package[-1])
            else:
                compute_package.append((1+t*(agrv_n))*compute_package[-1])
            #n取随机数
            
            agrv_n = np.random.uniform(0.899,0.925)
        
        length = len(cache_list)
        
        np.random.seed(3)
        print('*'*20)
        print('地域id:',area_num)
        print('*'*20)
        for t in range(length):
            while compute_package[-length:][t]>cache_list[t]*0.94:
                compute_package[t-length] = cache_list[t]*np.random.uniform(0.85,0.91)#超过预测值，改为预测值的90%左右
                
        cache_df = pd.DataFrame(columns=['areaid','date','predict_package','compute_package'])
        
        cache_df['date'] = pd.date_range('2017-07-01','2019-4-1',freq='M')
        cache_df['areaid'] = area_num
        cache_df['predict_package'] = cache_list
        cache_df['compute_package'] = compute_package
        cache_df['percent'] = cache_df['compute_package']/cache_df['predict_package']
        save_data = pd.concat([save_data,cache_df],ignore_index=True)
    print('结束时间：')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    end = time.time()
    print('*'*20)
    print('共用时：%s 秒'%(end-start))
    print('*'*20)
    return save_data

if __name__ == '__main__':
    #path = r"C:\Users\Sunsharp\Desktop\2019年工作日程\3mission\预测快递包裹数\chegndu_package.xlsx"
    #path = r"C:\Users\Sunsharp\Desktop\2019年工作日程\3mission\预测快递包裹数\compute_package_201903.xlsx"
    #path = input('请输入文件路径：')
    #sys.path.append(path)
    #file_name = r'chegndu_package.xlsx'
    predict_result = predict('package_data.xlsx')
    #print(predict_result.percent)
    predict_result.to_excel(r"package_predict_result_%s.xlsx"%time.strftime('%Y%m%d%H%M'))
    print("结果表为：package_predict_result_%s.xlsx"%time.strftime('%Y%m%d%H%M'))
    