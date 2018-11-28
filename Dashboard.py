########################Developed by Accenture#################################

import keras
from keras.layers import Input, Dense
from keras.models import Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
%matplotlib inline

from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(5)

dataset=pd.read_csv(r'C:\Users\ankit.c.khanna\Desktop\Project Documents\Adhoc work\Dash Dashboard\Updated code\input.csv')
dataset=dataset.fillna(0)
#dataset.columns
x_data= dataset.loc[:, ['1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M']]




#indicated the compression level of the data
encoding_dim = 6
#change the number to the number of input variables
input_img = Input(shape=(12,))
#defining encoding layer
encoded = Dense(encoding_dim, activation='relu')(input_img)
#defining decoding layer
decoded = Dense(12, activation='relu')(encoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the Autoencoder.It reconstructs the input data back with some reconstruction error
autoencoder.fit(x_data,x_data,
                epochs=100,
                batch_size=256,
                shuffle=True,validation_data=(x_data,x_data))



#**prediction 1				
#PREDICTING ON THE SAME DATA THAT WAS USED FOR FITTING TO ENSURE RECONSTRUCTION ERROR IS LESS			
predictions = autoencoder.predict(x_data)
df4=pd.DataFrame(predictions)
df4.columns = ['1_R_noerror', '2_R_noerror','3_R_noerror','4_R_noerror','5_R_noerror','6_R_noerror','7_R_noerror','8_R_noerror','9_R_noerror','10_R_noerror','11_R_noerror','12_R_noerror']
df4.tail()
#**prediction 1 ends#


###PREDICTING ON DATA WHICH HAS 10TH MONTH ANAMOLIES.ALL OTHER MONTHS DATA REMAIN INTACT.
#**prediction 2:
x_data_anom=dataset.loc[:, ['1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M_anomaly','11M','12M']]
predictions_anom = autoencoder.predict(x_data_anom)
df5=pd.DataFrame(predictions_anom)
df5.columns =  ['1M_R', '2_R','3_R','4_R','5_R','6_R','7_R','8_R','9_R','10_R','11_R','12_R']
df5.tail()

result=pd.DataFrame()
result = pd.concat([dataset[['UID','1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M_anomaly','11M','12M']],df5, df4], axis=1)
#renaming 10M_anomaly to 10M for plotting purpose
result.rename({'10M_anomaly':'10M'}, axis=1,inplace=True)
result.shape
result.head(2)

def reconstruct(actual,rec):
    return abs((actual-rec)/actual)


#result.columns

#result[result['2_R']-result['3_R']].ax
result['1M_RC']=result.apply(lambda x: reconstruct(x['1M'],x['1M_R']),axis=1)
result['2M_RC']=result.apply(lambda x: reconstruct(x['2M'],x['2_R']),axis=1)
result['3M_RC']=result.apply(lambda x: reconstruct(x['3M'],x['3_R']),axis=1)
result['4M_RC']=result.apply(lambda x: reconstruct(x['4M'],x['4_R']),axis=1)
result['5M_RC']=result.apply(lambda x: reconstruct(x['5M'],x['5_R']),axis=1)
result['6M_RC']=result.apply(lambda x: reconstruct(x['6M'],x['6_R']),axis=1)
result['7M_RC']=result.apply(lambda x: reconstruct(x['7M'],x['7_R']),axis=1)
result['8M_RC']=result.apply(lambda x: reconstruct(x['8M'],x['8_R']),axis=1)
result['9M_RC']=result.apply(lambda x: reconstruct(x['9M'],x['9_R']),axis=1)
result['10M_RC']=result.apply(lambda x: reconstruct(x['10M'],x['10_R']),axis=1)
result['11M_RC']=result.apply(lambda x: reconstruct(x['11M'],x['11_R']),axis=1)
result['12M_RC']=result.apply(lambda x: reconstruct(x['12M'],x['12_R']),axis=1)

def pred_score(Score,flag):
    if(Score>=flag ):
        return "Outlier"
    else:
        return "Inlier"
    

#Dropdown
features = ['1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M']



X= dataset.loc[:, ['UID','Product', 'Age_band','1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M']]

Product_options = []
for Product in X['Product'].unique():
    Product_options.append({'label':str(Product),'value':str(Product)})
#Product_options.append({'label':'All','value':'All'})

Age_band_options = []
for Age_band in X['Age_band'].unique():
    Age_band_options.append({'label':str(Age_band),'value':str(Age_band)})
#Age_band_options.append({'label':'All','value':'All'})

Q1_Data = pd.DataFrame(X.loc[:,['UID','Product','Age_band','1M', '2M','3M']])
Q1_Data['Avg_Balance'] = (X['1M'] + X['2M'] + X['3M'])/3

Q2_Data = pd.DataFrame(X.loc[:,['UID','Product','Age_band','4M', '5M','6M']])
Q2_Data['Avg_Balance'] = (X['4M'] + X['5M'] + X['6M'])/3

Q3_Data = pd.DataFrame(X.loc[:,['UID','Product','Age_band','6M', '7M','8M']])
Q3_Data['Avg_Balance'] = (X['6M'] + X['7M'] + X['8M'])/3

Q4_Data = pd.DataFrame(X.loc[:,['UID','Product','Age_band','10M', '11M','12M']])
Q4_Data['Avg_Balance'] = (X['10M'] + X['11M'] + X['12M'])/3


X_test= dataset.loc[:, ['1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M_anomaly','11M','12M']]
X_train=X_test










#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

###PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_features=pca.fit_transform(X_train_scaled)
print(pca.explained_variance_ratio_)

pca_test=pca.transform(X_test_scaled)

from sklearn.ensemble import IsolationForest
forest=IsolationForest(n_estimators=100,contamination=0.20 , max_samples=20,max_features=2,bootstrap ='True')
forestobj=forest.fit(pca_features)

t=forestobj.decision_function(pca_test)
pred=forestobj.predict(pca_test) 
X_new=np.c_[X_test,pred,t]
X_inline=X_new[X_new[:,3]==1]
X_outlie=X_new[X_new[:,3]==-1]
predcsv=pd.DataFrame(X_new)
predcsv['UID']=dataset['UID']



predcsv=pd.DataFrame(X_new)
predcsv['UID']=dataset['UID']


#12th month is named y axis and average of previous N months is X axis
#predcsv.columns = ['1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','Y-axis','Predictor','Score','UID']
predcsv.columns = ['1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M','Predictor','Score','UID']
#predcsv['X-axis']=predcsv[['1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M','11M']].mean(axis=1)
##calculating mean for y axis in the graph
predcsv['8M_mean']=predcsv[['1M', '2M','3M','4M','5M','6M','7M']].mean(axis=1)
predcsv['9M_mean']=predcsv[['1M', '2M','3M','4M','5M','6M','7M','8M']].mean(axis=1)
predcsv['10M_mean']=predcsv[['1M', '2M','3M','4M','5M','6M','7M','8M','9M']].mean(axis=1)
predcsv['11M_mean']=predcsv[['1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M']].mean(axis=1)
predcsv['12M_mean']=predcsv[['1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M','11M']].mean(axis=1)

predcsv=predcsv[['UID','1M', '2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M','Predictor','Score','8M_mean','9M_mean','10M_mean','11M_mean','12M_mean']]


def pred_score_IForest(Score,flag):
    if(Score>=flag ):
        return "Inlier"
    else:
        return "Outlier"
    






import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_table_experiments as dt
import dash_table
import dash_auth

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



app = dash.Dash()
server = app.server

from dash.dependencies import Input, Output
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.Div(
        html.H1('Data Quality Assessment - Anomaly Detection')
    ),
    dcc.Tabs(id="tabs", value='Tab1', children=[

        dcc.Tab(label='Distributions', id='tab1', value= 'Tab1', children=[
            html.Div(html.Label('Product'),style={'width':'10%','display': 'inline-block','textAlign': 'center','color': 'white',
                                                  'fontSize': 14,'background-color': 'MidnightBlue','font-weight': 'bold'}),
            html.Div(dcc.Dropdown(id='dropdown1',options=Product_options,value='home loans')),
            
            html.Div(html.Label('Age Band'),style={'width':'10%','display': 'inline-block','textAlign': 'center','color': 'white',
                                                  'fontSize': 14,'background-color': 'MidnightBlue','font-weight': 'bold'}),
            html.Div(dcc.Dropdown(id='dropdown2',options=Age_band_options,value='35-50')),
            html.Div(
                    dcc.Graph(id='Histogram1'),
                    style={'display': 'inline-block'}),
            html.Div(
                    dcc.Graph(id = 'Box-Plot1'),
                    style={'display': 'inline-block'}),
            html.Div(
                    dcc.Graph(id='Histogram2'),
                    style={'display': 'inline-block'}),
            html.Div(
                    dcc.Graph(id = 'Box-Plot2'),
                    style={'display': 'inline-block'}),
            html.Div(
                    dcc.Graph(id='Histogram3'),
                    style={'display': 'inline-block'}),
            html.Div(
                    dcc.Graph(id = 'Box-Plot3'),
                    style={'display': 'inline-block'}),
            html.Div(
                    dcc.Graph(id='Histogram4'),
                    style={'display': 'inline-block'}),
            html.Div(
                    dcc.Graph(id = 'Box-Plot4'),
                    style={'display': 'inline-block'})               
            
        ]),
        

            
        dcc.Tab(label ='Multivariate Outlier Detection', id='tab2', value='Tab2' ,children=[
                 html.Div([
         html.H1('Multivariate Outlier Detection-Isolation Forest')
        ]),
    
    
        html.Div([
        html.Div([
             html.Div('Choose Month', style={'color': 'white', 'fontSize': 14,'background-color': 'MidnightBlue',
                                             'font-weight': 'bold'}),
            dcc.Dropdown(
                id='xaxis1',
                options=[{'label': i, 'value': i} for i in features],
                value='10M'
            )
        ],
        style={'position': 'absolute', 'top': '200px', 'left': '10px','width': '200px','height':'40px', 'display': 'inline-block'}),
    

        
        html.Div([
            html.Div('Inlier', style={'color': 'white', 'fontSize': 14,'background-color': 'MidnightBlue',
                                          'text-align':'center','font-weight': 'bold'}),
                  #html.H1("Inlier"),
                 #html.P("Inlier"),
                 html.Div(id='inlier_tab1',style={'fontSize': 16,'font-weight': 'bold','text-align':'center'})
        ],
        style={'position': 'absolute', 'top': '200px', 'left': '500px', 'width': '150px',
               'height':'50px', 'display': 'inline-block','background-color': 'azure'}),
        
        html.Div([
            html.Div('Outlier', style={'color': 'white', 'fontSize': 14,'background-color': 'MidnightBlue',
                                          'text-align':'center','font-weight': 'bold'}),
                  #html.H1("Outlier"),
                 #html.P("Outlier"),
                 html.Div(id='outlier_tab1',style={'fontSize': 16,'font-weight': 'bold','text-align':'center'})
        ],
        style={'position': 'absolute', 'top': '200px', 'left': '700px', 'width': '150px',
               'height':'50px', 'display': 'inline-block', 'background-color': 'azure'})
    ]),
    
    html.Div([
           html.Div('Choose Threshold', style={'color': 'white', 'fontSize': 14,'text-align':'center',
                                               'background-color': 'MidnightBlue','font-weight': 'bold'}),
           dcc.Input(id='flag1',value='0',type='number',min='-1', max='1', step='0.05')],
        style={'position': 'absolute', 'top': '200px', 'left': '250px','width': '200px','height':'40px', 'display': 'inline-block'}),
        
   
    html.Div([
        dcc.Graph(id='feature-graphic1')],
        style={'position': 'absolute', 'top': '300px', 'left': '50px', 
               'width': '80%', 'display': 'inline-block','padding':10, 'vertical-align': 'middle'}),
            
    html.Div('na', style={'position':'absolute','color': 'white', 'fontSize': 1,
                                              'background-color': 'black','width': '100%','top': '760px','height':'5px'})   ,     
            
    html.Div([
   # generate_table(predcsv[['UID','1M','2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M']])   
       dash_table.DataTable(
        id='table',
           #predcsv['cal_pred_score']=='Outlier'
    columns=[{"name": i, "id": i} for i in predcsv[['UID','1M','2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M']].columns],
    data=predcsv[['UID','1M','2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M']].to_dict("rows"),
   )
    ],
     style={'position':'absolute','top': '780px', 'left': '50px','width': '90%',
            'border-collapse': 'collapse', 'display': 'inline-block','vertical-align': 'middle',
            'height': 300, 'overflowY': 'scroll'})
]),
                dcc.Tab(label='Deep Learning Based Anomaly Detection', id='tab3', value='Tab3', children =[
            html.Div([
         html.H1('Deep Learning Based Anomaly Detection-Autoencoders')
        ]),
    
    html.Div([
        html.Div([
             html.Div('Choose Month', style={'color': 'white', 'fontSize': 14,'background-color': 'MidnightBlue','font-weight': 'bold'}),
            dcc.Dropdown(
                id='xaxis',
                options=[{'label': i, 'value': i} for i in features],
                value='10M'
            )
        ],
        style={'position': 'absolute', 'top': '200px', 'left': '10px','width': '200px','height':'40px', 'display': 'inline-block'}),    
        
        html.Div([
                html.Div('Inlier', style={'color': 'white', 'fontSize': 14,'background-color': 'MidnightBlue',
                                          'text-align':'center','font-weight': 'bold'}),
                  #html.H1("Inlier"),
                 #html.P("Inlier"),
                 html.Div(id='inlier_tab',style={'fontSize': 16,'font-weight': 'bold','text-align':'center'})
        ],
        style={'position': 'absolute', 'top': '200px', 'left': '500px', 'width': '150px','height':'50px', 'display': 'inline-block','background-color': 'azure'}),
        html.Div([
            html.Div('Outlier', style={'color': 'white', 'fontSize': 14,'background-color': 'MidnightBlue',
                                       'text-align':'center','font-weight': 'bold'}),
                  #html.H1("Outlier"),
                 #html.P("Outlier"),
                 html.Div(id='outlier_tab',style={'fontSize': 16,'font-weight': 'bold','text-align':'center'})
        ],
        style={'position': 'absolute', 'top': '200px', 'left': '700px', 'width': '150px','height':'50px', 'display': 'inline-block', 'background-color': 'azure'})                
    ]),
      html.Div([
           html.Div('Choose Threshold', style={'color': 'white', 'fontSize': 14,'text-align':'center',
                                               'background-color': 'MidnightBlue','font-weight': 'bold'}),
           dcc.Input(id='flag',value='0.5',type='number',min='0', max='10', step='0.1')
           ],
        style={'position': 'absolute', 'top': '200px', 'left': '250px','width': '200px','height':'40px', 'display': 'inline-block'}),
                    
    html.Div([
    dcc.Graph(id='feature-graphic')],
        style={'position': 'absolute', 'top': '300px', 'left': '50px', 
               'width': '80%', 'display': 'inline-block','padding':10, 'vertical-align': 'middle'}),
            
    html.Div('na', style={'position':'absolute','color': 'white', 'fontSize': 1,
                'background-color': 'black','width': '100%','top': '760px','height':'5px'})   ,   
    
    html.Div([
    #generate_table(result[['UID','1M','2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M']]) 
        #dt.DataTable(id='datatable1',rows=result)
    ],
     style={'position':'absolute','top': '750px', 'left': '50px','width': '90%',
            'border-collapse': 'collapse', 'display': 'inline-block','vertical-align': 'middle',
            'height': 300, 'overflowY': 'scroll'})
            
    

        ])
#here
    ])
],style={'zoom':'90%'})




@app.callback(
    Output('feature-graphic', 'figure'),
    [Input('xaxis', 'value'),
    Input('flag', 'value')])


def update_graph(xaxis_name,flag):
    
    reconstructed_param=xaxis_name+"_RC"
    result['cal_pred_score']=result.apply(lambda x: pred_score(x[reconstructed_param],flag),axis=1)
    return {
            'data': [
                go.Scatter(
                    x=result[result['cal_pred_score'] == i][xaxis_name],
                    y=result[result['cal_pred_score'] == i][xaxis_name+"_RC"],
                    #text=result[result['cal_pred_score'] == i]['Score'],
                    mode='markers',
                    
                    marker={
                        'size': 15,
                        'opacity': 0.8,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in result.cal_pred_score.unique()
            ],
            'layout': go.Layout(
                
                xaxis=dict(title='Current Month Balance[$]'),
                yaxis=dict(title='%ge Difference from Predicted'),
                margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
 
    
@app.callback(
    Output('inlier_tab', 'children'),
    [Input('flag', 'value')])

def update_result_o(x):
    #return "The sum is: {}".format(x + y)
    inlier_count=result[result['cal_pred_score']=='Inlier']
    inlier_tab=inlier_count.shape[0]
    return inlier_tab


@app.callback(
    Output('outlier_tab', 'children'),
    [Input('flag', 'value')])

def update_result_i(x):
    #return "The sum is: {}".format(x + y)
    outlier_count=result[result['cal_pred_score']=='Outlier']
    outlier_tab=outlier_count.shape[0]
    return outlier_tab


@app.callback(Output('Histogram1', 'figure'),
              [Input('dropdown1', 'value'),
               Input('dropdown2', 'value')])

def update_histogram1(Product,Age_Band):
    return {
        'data': [go.Histogram(
            {'x': Q1_Data[(Q1_Data['Product'] == Product) & (Q1_Data['Age_band'] == Age_Band)]['Avg_Balance']},
            name='First Quarter',
            xbins=dict(start=-80000,
                       end=80000,
                       size=10000)
            )],
        'layout': go.Layout(
                title='First Quarter',
                xaxis=dict(title='Balance[Bin range:10K]'),
                yaxis=dict(title='#Customers'),
                bargap=0.2
               )     
    }

@app.callback(Output('Histogram2', 'figure'),
              [Input('dropdown1', 'value'),
               Input('dropdown2', 'value')])

def update_histogram2(Product,Age_Band):
    return {
        'data': [go.Histogram(
            {'x': Q2_Data[(Q2_Data['Product'] == Product) & (Q2_Data['Age_band'] == Age_Band)]['Avg_Balance']},
            name='Second Quarter',
            xbins=dict(start=-80000,
                       end=80000,
                       size=10000)
            )],
        'layout': go.Layout(
                title='Second Quarter',
                xaxis=dict(title='Balance[Bin range:10K]'),
                yaxis=dict(title='#Customers'),
                bargap=0.2
               )     
    }

@app.callback(Output('Histogram3', 'figure'),
              [Input('dropdown1', 'value'),
               Input('dropdown2', 'value')])

def update_histogram3(Product,Age_Band):
    return {
        'data': [go.Histogram(
            {'x': Q3_Data[(Q3_Data['Product'] == Product) & (Q3_Data['Age_band'] == Age_Band)]['Avg_Balance']},
            name='Third Quarter',
            xbins=dict(start=-80000,
                       end=80000,
                       size=10000)
            )],
        'layout': go.Layout(
                title='Third Quarter',
                xaxis=dict(title='Balance[Bin range:10K]'),
                yaxis=dict(title='#Customers'),
                bargap=0.2
               )     
    }

@app.callback(Output('Histogram4', 'figure'),
              [Input('dropdown1', 'value'),
               Input('dropdown2', 'value')])

def update_histogram4(Product,Age_Band):
    return {
        'data': [go.Histogram(
            {'x': Q4_Data[(Q4_Data['Product'] == Product) & (Q4_Data['Age_band'] == Age_Band)]['Avg_Balance']},
            name='Fourth Quarter',
            xbins=dict(start=-80000,
                       end=80000,
                       size=10000)
            )],
        'layout': go.Layout(
                title='Fourth Quarter',
                xaxis=dict(title='Balance[Bin range:10K]'),
                yaxis=dict(title='#Customers'),
                bargap=0.2
               )     
    }


@app.callback(Output('Box-Plot1', 'figure'),
              [Input('dropdown1', 'value'),
               Input('dropdown2', 'value')])

def BoxPlot1(Product,Age_Band):
    return {
        'data':[go.Box(
                {'y': Q1_Data[(Q1_Data['Product'] == Product) & (Q1_Data['Age_band'] == Age_Band)]['Avg_Balance']},
                                name = 'Average Quarterly Balance')],      
        'layout': go.Layout(
                                title='First Quarter')
            }


@app.callback(Output('Box-Plot2', 'figure'),
              [Input('dropdown1', 'value'),
               Input('dropdown2', 'value')])

def BoxPlot2(Product,Age_Band):
    return {
        'data':[go.Box({'y': Q2_Data[(Q2_Data['Product'] == Product) & (Q2_Data['Age_band'] == Age_Band)]['Avg_Balance']},
                                name = 'Average Quarterly Balance')],      
        'layout': go.Layout(
                                title='Second Quarter')
            }


@app.callback(Output('Box-Plot3', 'figure'),
              [Input('dropdown1', 'value'),
               Input('dropdown2', 'value')])

def BoxPlot3(Product,Age_Band):
    return {
        'data':[go.Box({'y': Q3_Data[(Q3_Data['Product'] == Product) & (Q3_Data['Age_band'] == Age_Band)]['Avg_Balance']},
                                name = 'Average Quarterly Balance')],      
        'layout': go.Layout(
                                title='Third Quarter')
            }


@app.callback(Output('Box-Plot4', 'figure'),
              [Input('dropdown1', 'value'),
               Input('dropdown2', 'value')])

def BoxPlot4(Product,Age_Band):
    return {
        'data':[go.Box({'y': Q4_Data[(Q4_Data['Product'] == Product) & (Q4_Data['Age_band'] == Age_Band)]['Avg_Balance']},
                                name = 'Average Quarterly Balance')],      
        'layout': go.Layout(
                                title='Fourth Quarter')
            }


@app.callback(
    Output('feature-graphic1', 'figure'),
    [Input('xaxis1', 'value'),
    Input('flag1', 'value')])


def update_graph(xaxis_name,flag):
    predcsv['cal_pred_score']=predcsv.apply(lambda x: pred_score_IForest(x['Score'],flag),axis=1)
    return {
            'data': [
                go.Scatter(
                    x=predcsv[predcsv['cal_pred_score'] == i][xaxis_name],
                    y=predcsv[predcsv['cal_pred_score'] == i][xaxis_name+"_mean"],
                    text=predcsv[predcsv['cal_pred_score'] == i]['Score'],
                    mode='markers',
                    
                    marker={
                        'size': 15,
                        'opacity': 0.8,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in predcsv.cal_pred_score.unique()
            ],
            'layout': go.Layout(
                xaxis=dict(title='Current Month Balance[$]'),
                yaxis=dict(title='Previous N Months Avg Balance[$]'),
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }

    
@app.callback(
    Output('inlier_tab1', 'children'),
    [Input('flag1', 'value')])

def update_result_o(x):
    #return "The sum is: {}".format(x + y)
    inlier_count=predcsv[predcsv['cal_pred_score']=='Inlier']
    inlier_tab=inlier_count.shape[0]
    return inlier_tab


@app.callback(
    Output('outlier_tab1', 'children'),
    [Input('flag1', 'value')])

def update_result_i(x):
    #return "The sum is: {}".format(x + y)
    outlier_count=predcsv[predcsv['cal_pred_score']=='Outlier']
    outlier_tab=outlier_count.shape[0]
    return outlier_tab


#########


if __name__ == '__main__':
    app.run_server()
    
    
    
    