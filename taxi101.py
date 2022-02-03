#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is for taxi101, please see requirement.txt and readme file.

@author:akhilesh.koul

"""

#library imports
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import geoplot as gplt
import json
import glob
from tqdm import tqdm
import xgboost as xgb
import pickle
from datetime import datetime, timedelta
from  time import time


from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support, plot_confusion_matrix, precision_score,recall_score,confusion_matrix, mean_absolute_error,mean_absolute_percentage_error,mean_squared_error

from shapely.geometry import Point
from scipy.spatial.distance import cdist
from geopy import distance
import pulp
import scipy




class Taxi101:
    

    def __init__(self,csv_path="data/taxi_chicago_2021.csv"):
        """
        Initialization function for the Taxi101 class. 
        
        
        Parameters
        ----------
        csv_path : STR, optional
            The path to read csv data file.The default is "data/taxi_chicago_2021.csv".

        Returns
        -------
        None.

        """
      
        
        self.csv_path=csv_path
     
    def readCsv(self,nrows=None):
        """
        Function to read the csv file

        Parameters
        ----------
        nrows : INT, optional
            The number of rows to read if need to sample the data, If None, all the rows are read. The default is None.

        Returns
        -------
        None.

        """
        
        start = time() 
        
        if nrows !=None:
            self.df = pd.read_csv(self.csv_path, nrows=nrows)
        if nrows == None:
            self.df = pd.read_csv(self.csv_path)    
        end = time()
        print("Time taken to load csv file : ",(end-start),"secs")



    def dataCleaning(self):
        """
        Function for data cleaning. Dropping NA values, and removing outliers etc.

        Returns
        -------
        None.

        """
    
        # print('Drop na values')    
        self.df_subset=self.df.dropna(subset=['Taxi ID','Fare','Trip Seconds','Trip Miles','Dropoff Centroid  Location','Pickup Centroid Location','Pickup Community Area','Dropoff Community Area'])
       

        # print("Selecting relevant Columns")        
        self.df_subset=self.df_subset[['Trip ID','Taxi ID','Trip Start Timestamp','Trip End Timestamp','Trip Seconds','Trip Miles','Pickup Community Area','Dropoff Community Area', 'Fare', 'Tips', 'Tolls', 'Extras','Payment Type','Company', 'Pickup Centroid Latitude','Pickup Centroid Longitude', 'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']]
        
        # print('Filtering Outliers using 3sigma rule')
        
        data_mean, data_std = np.mean(self.df_subset['Fare']), np.std(self.df_subset['Fare'])
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off

        mask=(self.df_subset['Fare'] < lower) | (self.df_subset['Fare'] > upper)
        self.df_subset['Outlier']=""
        self.df_subset.loc[mask,'Outlier']='1'
        self.df_subset.loc[~mask,'Outlier']='0'
        self.df_subset=self.df_subset.loc[~mask]


        # print('Removing 0 fare values')
        mask=(self.df_subset['Fare'].astype(int) ==0)
        self.df_subset= self.df_subset.loc[~mask]
        self.df_subset.reset_index(drop=True, inplace= True)
        
       
              
        cent_drop=100*((len(self.df)-len(self.df_subset))/(len(self.df)))
        
        print("Percentage Drop = " +str(round(cent_drop,2)) +" %")
    
        

        
    def featEng(self):
        """
        Function to do feature engineering.

        Returns
        -------
        None.

        """
        
        #print('Getting features from time')
        self.df_subset['Trip Start Timestamp'] = pd.to_datetime(self.df_subset['Trip Start Timestamp'],format='%m/%d/%Y %I:%M:%S %p')
        self.df_subset['Trip End Timestamp'] = pd.to_datetime(self.df_subset['Trip End Timestamp'],format='%m/%d/%Y %I:%M:%S %p')
        self.df_subset['start_time_hour']=self.df_subset['Trip Start Timestamp'].dt.hour
        self.df_subset['start_time_minute']=self.df_subset['Trip Start Timestamp'].dt.minute
        self.df_subset['start_time_day']=self.df_subset['Trip Start Timestamp'].dt.day
        self.df_subset['start_time_month']=self.df_subset['Trip Start Timestamp'].dt.month
        self.df_subset['start_time_weekday']=self.df_subset['Trip Start Timestamp'].dt.dayofweek
        self.df_subset['start_time_week']=self.df_subset['Trip Start Timestamp'].dt.isocalendar().week.astype(int)

        self.df_subset['end_time_hour']=self.df_subset['Trip End Timestamp'].dt.hour
        self.df_subset['end_time_minute']=self.df_subset['Trip End Timestamp'].dt.minute
        self.df_subset['end_time_day']=self.df_subset['Trip End Timestamp'].dt.day
        self.df_subset['end_time_month']=self.df_subset['Trip End Timestamp'].dt.month
        self.df_subset['end_time_weekday']=self.df_subset['Trip End Timestamp'].dt.dayofweek
        self.df_subset['end_time_week']=self.df_subset['Trip End Timestamp'].dt.isocalendar().week.astype(int)


        self.df_subset['Pickup Community Area']=self.df_subset['Pickup Community Area'].astype(int)
        self.df_subset['Dropoff Community Area']=self.df_subset['Dropoff Community Area'].astype(int)


        # print('Kmeans Clustering')
        #kmeans
        self.df_subset['kmeans']=""

        X=np.array(self.df_subset['Fare']).reshape(-1,1)
        self.kmeans_mdl = KMeans(n_clusters=3).fit(X)
        self.df_subset['kmeans']=self.kmeans_mdl.labels_
        
        self.df_subset['is_midnight']=""
        mask=((self.df_subset['start_time_hour']>=0) & (self.df_subset['start_time_hour']<6))
        self.df_subset.loc[mask,'is_midnight'] = 1
        self.df_subset.loc[~mask,'is_midnight'] = 0
        self.df_subset['is_midnight']=self.df_subset['is_midnight'].astype(int)
        
        #holiday
        # https://www.chicago.gov/city/en/narr/misc/city-holidays.html
        holidays=['01-01-2021', '18-01-2021', '12-02-2021', '15-02-2021', '01-03-2021', '31-05-2021', '05-07-2021', '06-09-2021', '11-10-2021', '25-11-2021', '24-12-2021', '31-12-2021']
        date_list= list(self.df_subset['Trip Start Timestamp'].dt.strftime('%d-%m-%Y'))
        is_holiday_list=[]

        for i in (range(len(date_list))):
            if date_list[i] in holidays:
                is_holiday_list.append(1)
            else:
                is_holiday_list.append(0)


        self.df_subset['is_holiday'] = is_holiday_list
        self.df_subset['is_holiday']=self.df_subset['is_holiday'].astype(int)


        pickup_gpd_df = gpd.GeoDataFrame()
        pickup_gpd_df['Trip ID']= self.df_subset['Trip ID']
        pickup_gpd_df['geometry']=[Point(xy) for xy in (zip( self.df_subset['Pickup Centroid Longitude'], self.df_subset['Pickup Centroid Latitude']))]
        
        dropoff_gpd_df = gpd.GeoDataFrame()
        dropoff_gpd_df['Trip ID']= self.df_subset['Trip ID']
        dropoff_gpd_df['geometry']=[Point(xy) for xy in (zip( self.df_subset['Dropoff Centroid Longitude'], self.df_subset['Dropoff Centroid Latitude']))]
        
        
        #geo_distance
        
        distance_list=[]
        for i in tqdm(range(len(self.df_subset))):
            A = (pickup_gpd_df['geometry'][i].x,pickup_gpd_df['geometry'][i].y)
            B = (dropoff_gpd_df['geometry'][i].x,dropoff_gpd_df['geometry'][i].y)
            distance_list.append(distance.great_circle(A,B).miles)
        
        self.df_subset['geoDistance']=distance_list
        
    
    def pie_chart(self,pie_type ="size", groupby_col="Company", groupby_cols=None, sum_col='Fare', agg_cmd={"Taxi ID": "nunique"}, title ="Total Market Share 2021", plot=True):
        """
        Function to plot pie chart and get the respective stats.For detail explanation see matplotlib documentation.

        Parameters
        ----------
        pie_type : STR, optional
            'size' for getting number stats.'sum' for getting summation stats, 'custom' for custom grouping
        groupby_col : STR, optional
            Column to be used for grouping..
        groupby_cols : STR, optional
            If more than one column to be used for grouping. Use columns as list. 
        sum_col : STR, optional
            Column over which summation is to be performed. The default is 'Fare'.
        agg_cmd : DICT, optional
            Used for Custom grouping,The default is {"Taxi ID": "nunique"}.
        title : STR, optional
            Title of the plot. The default is "Total Market Share 2021".
        plot :BOOL, optional
            To plot or not . The default is True.

        Returns
        -------
        pie_data : PANDAS DATAFRAME or PANDAS SERIES
            Grouped dataframe or series.

        """
        
            
        if pie_type != 'custom':
            if pie_type=="size":
                pie_data=self.df_subset.groupby([groupby_col]).size()
                
                if groupby_col !='kmeans':
                    pie_data=pie_data.sort_values()
                # print(pie_data)
            
            if pie_type =="sum":
                pie_data=self.df_subset.groupby([groupby_col]).sum()[sum_col]
                
                if groupby_col !='kmeans':
                    pie_data=pie_data.sort_values()
                # print(pie_data)
            
            if groupby_col =='kmeans':
                pie_label=list(zip(list(pie_data.index),self.kmeans_mdl.cluster_centers_.round().flatten()))
                # print(pie_label)
            
            if groupby_col !='kmeans':
                pie_label=list(pie_data.index) 
                # print(pie_label)
            
            if plot == True:
                fig = plt.subplots(figsize=(15,8))
                # if groupby_col !='kmeans':
                plt.pie(x=pie_data,labels=pie_label,autopct='%.0f%%',shadow=True)
                plt.title(title)
                # plt.legend()
                # plt.legend(bbox_to_anchor=(1,0), loc="upper left")
                plt.show()
                


        if pie_type == 'custom':
            
            if groupby_cols != None :
                pie_data=self.df_subset.groupby(groupby_cols)
                pie_data = (pie_data.agg(agg_cmd))
                pie_data= pie_data.droplevel(1)
                pie_data[groupby_col]= pie_data.index
                pie_data.reset_index(drop=True, inplace= True)
                pie_data =pie_data.groupby([groupby_col]).count()
                pie_data=pie_data.sort_values(by=list(agg_cmd.keys())[0])
        
            if groupby_cols==None :
                
                pie_data=self.df_subset.groupby([groupby_col])
                pie_data = pie_data.agg(agg_cmd)
                pie_data.reset_index(drop=True, inplace= True)
                pie_data =pie_data.value_counts()
                pie_data=pie_data.sort_values()
            
            if plot == True:
                fig = plt.subplots(figsize=(15,8))
                plt.pie(x=pie_data.values.flatten(),labels=list(pie_data.index),autopct='%.0f%%',shadow=True)
                plt.title(title)
                # plt.legend()
                # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.show()
        
        return pie_data
        
    def kde_plot(self,x_col,hue_col=None,x_label=None,title=None):
        """
        Function to plot Kernel Density Estimation. For detailed explanation see seaborn documentation.

        Parameters
        ----------
        x_col : STR
            X_axis column name for which KDE is to be plotted.
        hue_col : STR, optional
            Column name for which hue is to be plotted. The default is None.
        
        x_label : STR, optional
            Custom x-axis label name for the plot. The default is None.
        title : STR, optional
            Title of the plot. The default is None.

        Returns
        -------
        None.

        """
            
        fig,ax=plt.subplots()
        for x_c in x_col:
            sns.kdeplot(data=self.df_subset,x=x_c,hue=hue_col,ax=ax,label=x_c)
            
        if len(x_col)>1:
            plt.xlabel(x_label)
            plt.legend()
            
            
        # plt.legend()
        plt.title(title)
        plt.show()
        
    def box_plot(self,x_col='kmeans',y_col=None,groupby_col=['Taxi ID','kmeans'],box_type ='sum',sum_col='Fare', ratio=1,plot=True,title=None):
        """
        Function to plot box-plot and get the respective statistics.
        For detail explanation see seaborn documentation.

        Parameters
        ----------
        x_col : STR, optional
            Column name for x-axis. The default is 'kmeans'.
        y_col : STR, optional
            Column Name for y-axis. The default is None.
        groupby_col : STR, optional
            Column name if the data is to be grouped. The default is ['Taxi ID','kmeans'].
        box_type : STR, optional
            'size' to get occurrence stats, 'sum' to get summation stats. The default is 'sum'.
        sum_col : STR, optional
            Column Name to be used in summation. The default is 'Fare'.
        ratio : INT, optional
            Ratio to used if need to analysis for different scale, e.g. 365 for daily stats. The default is 1.
        plot : BOOL, optional
            True if plot is required. The default is True.
        title : STR, optional
            Title of the plot. The default is None.

        Returns
        -------
        box_data : PANDAS DATAFRAME or PANDAS SERIES
            Box data stats as dataframe or series.

        """
            
        if box_type == 'sum':
            box_data=pd.DataFrame(self.df_subset.groupby(groupby_col).sum()[sum_col]/ratio)
            box_data[x_col]=np.array(box_data.index.get_level_values(x_col))
            box_data.reset_index(drop=True,inplace=True)
        
        if box_type == 'size':
            box_data=pd.DataFrame(self.df_subset.groupby(groupby_col).size()/ratio,columns=['trips'])
            box_data[x_col]=np.array(box_data.index.get_level_values(x_col))
            box_data.reset_index(drop=True,inplace=True)
            y_col='trips'
            
        if plot == True:    
            sns.boxplot(x=x_col,y=y_col,data=box_data)   
            plt.title(title)
            plt.show()   
        
        return box_data

    def corrd_to_area(self,flag='train',long=None,lat=None):
        """
        Function to obtain Area number with co-ordinates as input

        Parameters
        ----------
        flag : STR, optional
            'train' is training is to be done. 'test' to use trained model. The default is 'train'.
        long : FLOAT, optional
            Longitude. The default is None.
        lat : FLOAT, optional
            Latitude. The default is None.

        Returns
        -------
        area : INT
            Area Number.

        """
       
        if flag == 'train':
            pick_loc=pd.DataFrame()
            pick_loc['Area']=self.df_subset['Pickup Community Area']
            pick_loc['long']=self.df_subset['Pickup Centroid Longitude']
            pick_loc['lat']=self.df_subset['Pickup Centroid Latitude']
            
            
            drop_loc=pd.DataFrame()
            drop_loc['Area']=self.df_subset['Dropoff Community Area']
            drop_loc['long']=self.df_subset['Dropoff Centroid Longitude']
            drop_loc['lat']=self.df_subset['Dropoff Centroid Latitude']
            
         
            frames = [pick_loc , drop_loc]
            result_loc = pd.concat(frames)
            result_loc.reset_index(drop=True, inplace=True)
            
            X=np.array(result_loc[['long','lat']])
            y=np.array(result_loc['Area'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            neigh = KNeighborsClassifier(algorithm='kd_tree',n_neighbors=77,n_jobs=-1,leaf_size=4)
            neigh.fit(X_train, y_train)
            
            pickle.dump(neigh, open('data/neigh.pkl','wb'))
        
          
            y_test_true=y_test[0:100]
            y_pred_test=neigh.predict(X_test[0:100])
            # print(f1_score(y_test_true, y_pred_test, average='micro'))
           
            return None
           
        if flag=='test':
             neigh_load = pickle.load(open('data/neigh.pkl', 'rb'))
             X=np.array([long,lat]).reshape(1,-1)
             area=neigh_load.predict(X).item()
             # print(area)
             return area

    def geoSpatialArea(self,area_type='Pickup'):
        """
        Function to visualize data  pickup or dropoff are in geospatial sense or on Map.

        Parameters
        ----------
        area_type : STR, optional
            'Pickup' for pickup area, 'Dropoff' for dropoff area. The default is 'Pickup'.

        Returns
        -------
        None.

        """

        shp_file=glob.glob('data/Boundaries - Community Areas (current)/*.shp')[0]
        city_boundary = gpd.read_file(shp_file)


        long_centroid = area_type+ " Centroid Longitude"
        lat_centroid = area_type+ " Centroid Latitude"
        
        df_subset_tmp=self.df_subset[['Trip ID',long_centroid,lat_centroid]]
        df_group=df_subset_tmp.groupby([long_centroid,lat_centroid]).count() 
        df_group['long']=df_group.index.get_level_values(long_centroid)
        df_group['lat']=df_group.index.get_level_values(lat_centroid)
        df_group.reset_index(drop=True,inplace=True)

        geo_df=gpd.GeoDataFrame()
        geometry=[Point(xy) for xy in (zip(df_group['long'],df_group['lat']))]
        geo_df['geometry']=geometry
        geo_df['Count']=np.log(df_group['Trip ID'])
        ax=gplt.pointplot(geo_df, cmap='inferno_r', projection=gplt.crs.AlbersEqualArea(),
                             hue='Count', legend=True, scale='Count',legend_var='hue')  
                
        gplt.polyplot(city_boundary, projection=gplt.crs.AlbersEqualArea(),zorder=1,ax=ax)
        plt.title(area_type)
        plt.show()
    
    def xgb_price_train(self):
        """
        Function to train regression model for Price prediction


        Returns
        -------
        None.

        """       
                
        X=self.df_subset[['Pickup Community Area','Dropoff Community Area','Pickup Centroid Latitude','Pickup Centroid Longitude','Dropoff Centroid Latitude','Dropoff Centroid Longitude', 'start_time_hour','start_time_minute','start_time_day','start_time_month','start_time_weekday', 'start_time_week','is_midnight','is_holiday','geoDistance']]
            
        y=self.df_subset['Fare']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=42)
       
        self.X_test=X_test
        self.y_test=y_test
            
        reg=xgb.XGBRegressor(objective='reg:squarederror',eta= 0.1, max_depth=8,n_estimators=1000,random_state=42)
        reg.fit(X_train,y_train,eval_set=[(X_val, y_val)], verbose=100,early_stopping_rounds=100)

        pickle.dump(reg, open('data/xgb.pkl','wb'))
    
        y_pred_test=reg.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print('rmse = '+str(rmse))
        mae = mean_absolute_error(y_test, y_pred_test)
        print('mae = '+str(mae))
        return None
    
    def plot_corr(self):
        """
        Function to plot correlation data with 'Fare'

        Returns
        -------
        None.

        """
        
        fig, ax = plt.subplots(figsize=(6,6))
        
        df_corr=self.df_subset[['Pickup Community Area','Dropoff Community Area','Pickup Centroid Latitude','Pickup Centroid Longitude','Dropoff Centroid Latitude','Dropoff Centroid Longitude', 'start_time_hour','start_time_minute','start_time_day','start_time_month','start_time_weekday', 'start_time_week','is_midnight','is_holiday','geoDistance','Fare']]
      
        
        sns.heatmap(df_corr.corr()[['Fare']].sort_values('Fare'), vmax=1, vmin=-1, cmap='YlGnBu', annot=True, ax=ax)
        ax.invert_yaxis()
        plt.show()


    
    def price_is_right(self, X=None,y_test=False, multiplier=1.0):
        """
        Price engine function on already trained model

        Parameters
        ----------
        X : PANDAS DATAFRAME, optional
            Input feature for regression model. The default is None.
        y_test : Bool, optional
            'True' if want to see the actual 'Fare' value in the test set. The default is False.
        multiplier : INT, optional
            To be used as a scale when there is upsurge demand or downtrend demand. The default is 1.0.

        Returns
        -------
        FLOAT
            Recommended Price.

        """
        
        reg_load = pickle.load(open('data/xgb.pkl', 'rb'))
        
        journey_df=gpd.GeoDataFrame()
        shp_file=glob.glob('data/Boundaries - Community Areas (current)/*.shp')[0]
        city_boundary = gpd.read_file(shp_file)
        to_from=['Pickup','Dropoff']
        area=[]
        geo=[]
        for j in to_from:
            area.append(j)
            long=j +' Centroid Longitude'
            lat=j +' Centroid Latitude'
            geo.append(Point(X[long],X[lat]))
        
        journey_df['journey']=area
        journey_df['geometry']=geo
          
        ax=gplt.pointplot(journey_df, cmap='seismic', projection=gplt.crs.AlbersEqualArea(),hue='journey', legend=True,legend_var='hue',linewidth=10)     
        gplt.polyplot(city_boundary, projection=gplt.crs.AlbersEqualArea(),zorder=1,ax=ax)
         
        plt.show()
        
     
        y_pred=round(reg_load.predict(np.array(X).reshape(1,-1)).item(),1)
        
     
       
        if y_test==True:
            print('Actual Price = ' + str(self.y_test[X.index]))
        
        if multiplier !=1.0:
            print('The Multiplier for the current time is :' +str(multiplier))
     
        return multiplier*y_pred
    
    def linear_prog(self,upper_bound='q3'):
        """
        Fnction to do linear programming to find optimal revenue

        Parameters
        ----------
        upper_bound : STR, optional
            Stat type, 'q3' for Q3 quantile, 'mean'  and 'median'. The default is 'q3'.

        Returns
        -------
        None.

        """

        
        kmeans=[[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
        bound=[]

        for n in range(len(kmeans)):
            df_train_filtered_df = self.df_subset[self.df_subset['kmeans'].isin(kmeans[n])]
             
            group_data=pd.DataFrame(df_train_filtered_df.groupby(['Taxi ID']).size(),columns=['trips'])
        
            group_data.reset_index(drop=True,inplace=True)
            
            X_trip = np.array(group_data[['trips']])
            
            if upper_bound=='q3':
                bound.append([np.percentile(X_trip, 25, interpolation = 'midpoint'),np.percentile(X_trip, 75, interpolation = 'midpoint')])
                
            if upper_bound =='median':
                bound.append([np.percentile(X_trip, 25, interpolation = 'midpoint'),np.median(X_trip)])    
            
            if upper_bound =='mean':
                  bound.append([np.percentile(X_trip, 25, interpolation = 'midpoint'),np.mean(X_trip)])    
              
      
        #linear programming
        model = pulp.LpProblem('Taxi101', sense= pulp.LpMaximize)
        
        k0 = pulp.LpVariable('k0', lowBound=0, upBound=None, cat='Integer')
        k1 = pulp.LpVariable('k1', lowBound=0, upBound=None, cat='Integer')
        k2 = pulp.LpVariable('k2', lowBound=0, upBound=None, cat='Integer')

        model += round(self.kmeans_mdl.cluster_centers_[0].item(),1) * k0 + round(self.kmeans_mdl.cluster_centers_[1].item(),1) * k1 + round(self.kmeans_mdl.cluster_centers_[2].item(),1) * k2
        
        model += k0  >= bound[0][0]
        model += k0  <= bound[0][1]
        model += k1  >= bound[1][0]
        model += k1  <= bound[1][1]
        model += k2  >= bound[2][0]
        model += k2  <= bound[2][1]

        model += k0 + k1 >= bound[3][0]
        model += k0 + k1 <= bound[3][1]

        model += k0 + k2 >= bound[4][0]
        model += k0 + k2 <= bound[4][1]

        model += k1 + k2 >= bound[5][0]
        model += k1 + k2 <= bound[5][1]

        model += k0 + k1 + k2 >= bound[6][0]
        model += k0 + k1 + k2 <= bound[6][1]
        
        model.solve(pulp.COIN_CMD(msg=False))
        self.optimal_value=pulp.value(model.objective)

     
    def market_launch(self,percent,upper_bound='q3',title= None):
        """
        Function to analuse market reveince and find the market potentoin with optimal value.

        Parameters
        ----------
        percent : INT
            Percentage value [0,100], percentage of the taxi market share 
        upper_bound : STR, optional
            Stat type, 'q3' for Q3 quantile, 'mean'  and 'median'. The default is 'q3'.
        title : STR, optional
            Plot title. The default is None.

        Returns
        -------
        LIST
            List with optimal market revenye and/or exsiting market reenebce.

        """


        
        self.linear_prog(upper_bound=upper_bound)
        
        kmeans=[[0],[1],[2]]
        rv=[]
        
        for k in range(len(kmeans)):
            # print(k)
        
            df_train_filtered_df = self.df_subset[self.df_subset['kmeans'].isin(kmeans[k])]
             
            group_data=pd.DataFrame(df_train_filtered_df.groupby(['Taxi ID']).size(),columns=['trips'])

            group_data.reset_index(drop=True,inplace=True)
            X_trip = np.array(group_data[['trips']])
           
            k_dist = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_trip)
            k_rv=(k_dist.sample(n_samples=2340,random_state=42)).flatten()
            rv.append(k_rv)
            # fig,ax=plt.subplots()
            # sns.kdeplot(np.array(X_trip).flatten(),ax=ax)
            # sns.kdeplot(k_rv,ax=ax)
            # plt.show()
   

        if percent == None:
   
            value1=2340*(round(self.kmeans_mdl.cluster_centers_[0].item(),1)  * rv[0]+  round(self.kmeans_mdl.cluster_centers_[1].item(),1)  * rv[1] +  round(self.kmeans_mdl.cluster_centers_[2].item(),1)  * rv[2])
        
            sns.histplot(value1,label='Existing distribution revenue',kde=True)
            plt.axvline(x=self.df_subset['Fare'].sum(),color='g',linestyle='-.',label='Current revenue')
            total_optimal_value=2340*(self.optimal_value)
            plt.axvline(x=total_optimal_value, color='r', linestyle='-.',label='Optimal revenue')
            plt.yticks([],[])
            plt.ylabel("")
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            plt.title(title)
            plt.legend()
            plt.show()
            return self.df_subset['Fare'].sum(),total_optimal_value
        
        if percent != None:
            
            taxi_number=int((percent*2340)/100)
        
            value2=taxi_number*(round(self.kmeans_mdl.cluster_centers_[0].item(),1)  * rv[0]+  round(self.kmeans_mdl.cluster_centers_[1].item(),1)  * rv[1] +  round(self.kmeans_mdl.cluster_centers_[2].item(),1)  * rv[2])
    
            sns.histplot(value2,label='Existing distribution revenue' ,kde=True)
            total_optimal_value=450*(self.optimal_value)
            plt.axvline(x=total_optimal_value, color='r', linestyle='-.',label='Optimal revenue')
            plt.yticks([],[])
            plt.ylabel("")
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            plt.title(title)
            plt.legend()
            plt.show()
            return total_optimal_value
            
      
        
            



if __name__ == '__main__':
    print('Happy')
    taxi=Taxi101()
    taxi.readCsv(nrows=10000)
    taxi.dataCleaning()
    taxi.featEng()
    
    # #company share count
    # _=taxi.pie_chart(pie_type ="size", groupby_col="Company", title ="Company wise Market Size 2021 - Numbers")
    
    # #company share sum
    # _=taxi.pie_chart(pie_type ="sum", groupby_col="Company", sum_col='Fare', title ="Company wise Market Size 2021 - Revenue" )
    
    #  #company_taxi
    # _=taxi.pie_chart(pie_type ="custom", groupby_col="Company", groupby_cols=['Company','Taxi ID'], sum_col='Fare', agg_cmd={"Taxi ID": "nunique"}, title ="Company wise Tax ownership" )
     
    # #taxi to company mapping
    # _=taxi.pie_chart(pie_type ="custom", groupby_col="Taxi ID", agg_cmd={"Company": "nunique"}, title ="Taxi wise Loyality" )

    #    #kmeans sum
    # _=taxi.pie_chart(pie_type ="sum", groupby_col="kmeans", groupby_cols=None, sum_col='Fare', title ="Kmeans Sum" )
              
    #  #kmeans size
    # _=taxi.pie_chart(pie_type ="size", groupby_col="kmeans", groupby_cols=None, sum_col='Fare', title ="Kmeans Sum" ) 
    
    
    # #pickup location sum
    # _=taxi.pie_chart(pie_type ="sum", groupby_col="Pickup Community Area", sum_col='Fare', title ="Pickup COmmnity area Market Revenue-2021")

    # #dropoff location sum
    # _=taxi.pie_chart(pie_type ="sum", groupby_col="Dropoff Community Area", sum_col='Fare', title ="Pickup COmmnity area Market Revenue-2021")
    
    
    # #payment_type size
    # _=taxi.pie_chart(pie_type ="size", groupby_col="Payment Type", title ="Payment Type Size")

    # #payment_type sum
    # _=taxi.pie_chart(pie_type ="sum", groupby_col="Payment Type",sum_col="Fare", title ="Payment Type Sum")

    #  #weekend sum
    # _=taxi.pie_chart(pie_type ="sum", groupby_col="start_time_weekday",sum_col="Fare", title ="start_time_weekday")  
     
     
    # #weekend size
    # _=taxi.pie_chart(pie_type ="size", groupby_col="start_time_weekday", title ="start_time_weekday" )
      
    # #is_midnight sum
    # _=taxi.pie_chart(pie_type ="sum", groupby_col="is_midnight",sum_col="Fare", title ="is_midnight")

    # #is_midnight size
    # _=taxi.pie_chart(pie_type ="size", groupby_col="is_midnight", title ="is_midnight" )
    
    # #is_holiday sum
    # _=taxi.pie_chart(pie_type ="sum", groupby_col="is_holiday",sum_col="Fare", title ="is_holiday")

    # #is_holiday size
    # _=taxi.pie_chart(pie_type ="size", groupby_col="is_holiday", title ="is_holiday" )

    # taxi.kde_plot(x_col=['Fare'])  
    
    # taxi.kde_plot(x_col=['Fare'],hue_col='kmeans')   
    
    # taxi.kde_plot(x_col=['Pickup Community Area','Dropoff Community Area'], x_label='Community Area')   
    
    
    # #with kmeans
    # taxi.kde_plot(x_col=["Pickup Community Area"],hue_col='kmeans')
    # taxi.kde_plot(x_col=["Dropoff Community Area"],hue_col='kmeans')
    
    # #pickup weekday
    # taxi.kde_plot(x_col=["Pickup Community Area"],hue_col='start_time_weekday')
    
    # #dropoff weekday
    # taxi.kde_plot(x_col=["Dropoff Community Area"],hue_col='start_time_weekday')
    
    # #payment Type
    # taxi.kde_plot(x_col=["Fare"],hue_col='Payment Type')
    
    # #is_midnight kde
    # taxi.kde_plot(x_col=["Fare"],hue_col='is_midnight')
    
    # #holiday kde
    # taxi.kde_plot(x_col=["Fare"],hue_col='is_holiday')
   

    # taxi.box_plot(x_col='kmeans',y_col='Fare',groupby_col=['Taxi ID','kmeans'], box_type ='sum', sum_col='Fare', ratio=1)
    # taxi.box_plot(x_col='kmeans',y_col='Fare',groupby_col=['Taxi ID','kmeans'], box_type ='sum', sum_col='Fare', ratio=365)
           
    # taxi.box_plot(x_col='kmeans', groupby_col=['Taxi ID','kmeans'],box_type ='size',ratio=1)
    # taxi.box_plot(x_col='kmeans', groupby_col=['Taxi ID','kmeans'],box_type ='size',ratio=365)
                   
    

    # taxi.corrd_to_area(flag='train')
    # taxi.corrd_to_area(flag='test',long=-87.633308,lat=41.899602)
           
    # taxi.geoSpatialArea(area_type='Pickup')
    # taxi.geoSpatialArea(area_type='Dropoff')

    # taxi.xgb_price_train(flag='train') 
    
    # taxi.plot_corr()
    # taxi.price_is_right(X=taxi.X_test.sample(),multiplier=1.0)



    # _=taxi.market_launch(percent=None,upper_bound='q3',title= 'Market Analysis with upper bound as Q3')
    # print("Current Revenue " + "${:,.2f}".format(_[0]))
    # print("Optimal Revenue with upper bound as Q3 Revenue " + "${:,.2f}".format(_[1]))
    
    # _=taxi.market_launch(percent=None,upper_bound='mean',title= 'Market Analysis with upper bound as Mean Value')
    # print("Current Revenue " + "${:,.2f}".format(_[0]))
    # print("Optimal Revenue with upper bound as Mean Revenue " + "${:,.2f}".format(_[1]))
    
    # _=taxi.market_launch(percent=None,upper_bound='median',title= 'Market Analysis with upper bound as Median Value')
    # print("Current Revenue " + "${:,.2f}".format(_[0]))
    # print("Optimal Revenue with upper bound as Median Revenue " + "${:,.2f}".format(_[1]))
    
    # _=taxi.market_launch(percent=10,upper_bound='q3',title= 'Market Analysis with upper bound as Q3 and Taxi-Market share of 10%')
    # print("Optimal Revenue with upper bound as Q3 Revenue and Taxi-Market share of 10% : " + "${:,.2f}".format(_))
    
    # _=taxi.market_launch(percent=10,upper_bound='mean',title= 'Market Analysis with upper bound as Mean Value and Taxi-Market share of 10%')
    # print("Optimal Revenue with upper bound as Mean Revenue and Taxi-Market share of 10% : " + "${:,.2f}".format(_))
    
    # _=taxi.market_launch(percent=10,upper_bound='median',title= 'Market Analysis with upper bound as Median Value and Taxi-Market share of 10% ')

    # print("Optimal Revenue with upper bound as Median Revenue and Taxi-Market share of 10% : " + "${:,.2f}".format(_))
    

         


