import tensorflow as tf
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
from keras.utils.vis_utils import plot_model
from string import ascii_uppercase
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DataPrep:
    MTD_url = "C:/Users/matth/Desktop/MTD-NN/MeltTraceDatabase.csv"
    Columns_url = "C:/Users/matth/Desktop/MTD-NN/ColumnsIN.csv" 

    def LoadData(MTD_url=MTD_url,Columns_url=Columns_url):
        df = pd.read_csv(MTD_url,dtype=str,encoding='cp1252')
        df = df.fillna(0, downcast='infer').astype(str)

        keptcsv = pd.read_csv(Columns_url,encoding='utf8').astype(str)
        keptvars = list(keptcsv.columns.values)
        #print(keptvars)
        df.drop(df.columns.difference(keptvars), 1, inplace=True)
        #print(df.shape)
        #print(df.head())
        df.drop(df.index[0:4300], axis=0, inplace=True)

        for column in df:
            df[column] = df[column].str.replace("%","")
            df[column] = df[column].str.replace("!","0")
            #print(df[column])
        
        return df  

    def dataframe_to_dataset(df):
        df = df.copy()
        dftarget = df.iloc[:, 37:64]
        dfdata = df.iloc[:,:37]
        ds = tf.data.Dataset.from_tensor_slices((dict(dfdata), dftarget))
        ds = ds.shuffle(buffer_size=len(df))
        return ds

    def convert_MTD_usefull(df):
        d = {'1': [0],'2':[0],'3': [0],'4': [0],'5': [0],'6': [0],'7': [0],'8': [0],'9': [0],'A': [0],'B': [0],'C': [0],'D': [0]
            ,'E': [0],'F': [0],'G': [0],'H': [0],'I': [0],'J': [0],'K': [0],'L': [0],'M': [0],'N': [0],'O': [0],'P': [0],'Q': [0]
            ,'Z': [0]
        }
        dfCON = pd.DataFrame(data=d)
        dfnew = pd.concat([df,dfCON],axis=0)
        dfnew = dfnew.drop([0]).fillna(0, downcast='infer').astype(str)
        #dfnew.head()
        
        for index, row in dfnew.iterrows():
            burnkey = str(dfnew.at[index,"Burnkey"])
            #print(burnkey)
            for c in ascii_uppercase:
                #print(c)
                if str(c) in burnkey:
                    dfnew.at[index, c] = '1'
            for i in range(1,9):
                #print(i)
                if str(i) in burnkey:
                    dfnew.at[index, str(i)] = '1'
        return dfnew
    
    def encode_numerical_feature(feature, name, dataset):
        # Create a Normalization layer for our feature
        normalizer = Normalization()

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the statistics of the data
        normalizer.adapt(feature_ds)

        # Normalize the input feature
        encoded_feature = normalizer(feature)
        return encoded_feature

    def encode_categorical_feature(feature, name, dataset, is_string):
        lookup_class = StringLookup if is_string else IntegerLookup
        # Create a lookup layer which will turn strings into integer indices
        lookup = lookup_class(output_mode="binary")

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the set of possible string values and assign them a fixed integer index
        lookup.adapt(feature_ds)

        # Turn the string input into integer indices
        encoded_feature = lookup(feature)
        return encoded_feature

    def batch_data():
        df = DataPrep.LoadData()
        df = DataPrep.convert_MTD_usefull(df)

        for column in df:
            if not(column == "Burnkey" or column == "BurnComment"):
                df[column] = df[column].astype(float)
            else:
                df.drop([column],  1, inplace=True)

        val_dataframe = df.sample(frac=0.2, random_state=1337)
        train_dataframe = df.drop(val_dataframe.index)
        
        print(
            "Using %d samples for training and %d for validation"
            % (len(train_dataframe), len(val_dataframe))
        )   

        train_ds = DataPrep.dataframe_to_dataset(train_dataframe)
        val_ds = DataPrep.dataframe_to_dataset(val_dataframe)

        # for x, y in train_ds.take(1):
        #     print("Input:", x)
        #     print("Target:", y)

        train_ds = train_ds.batch(64)
        val_ds = val_ds.batch(64)
        sample_ds = train_ds

        return train_ds, val_ds, sample_ds


train_ds, val_ds, sample_ds = DataPrep.batch_data()


# input = keras.Input(shape=(37,),name="all")

# Numerical features
Target_Al = keras.Input(shape=(1,))
Target_Lime = keras.Input(shape=(1,))
Target_Fluorspar = keras.Input(shape=(1,))
Pred_Heat = keras.Input(shape=(1,))
Mix_Weight = keras.Input(shape=(1,))
RMC_Weight = keras.Input(shape=(1,))
Mo_lbs = keras.Input(shape=(1,))
Mo_lbs_with_DCE_and_Remelt = keras.Input(shape=(1,))
Mo_percent = keras.Input(shape=(1,))
MoO3 = keras.Input(shape=(1,))
MoO2 = keras.Input(shape=(1,))
S_percent_RMC = keras.Input(shape=(1,))
Ca_percent_RMC = keras.Input(shape=(1,))
Na_percent_RMC = keras.Input(shape=(1,))
Total_AcIn = keras.Input(shape=(1,))
Si_kMol = keras.Input(shape=(1,))
Mo_kMol = keras.Input(shape=(1,))
MoO3_kMol = keras.Input(shape=(1,))
MoO2_kMol = keras.Input(shape=(1,))
Fe3O4_kMol = keras.Input(shape=(1,))
Al_kMol = keras.Input(shape=(1,))
Mg_kMol = keras.Input(shape=(1,))
Ca_m_kMol = keras.Input(shape=(1,))
Ca_f_kMol = keras.Input(shape=(1,))
Base_Lime = keras.Input(shape=(1,))
Ca_RMC_kg = keras.Input(shape=(1,))
Ca_DCE = keras.Input(shape=(1,))
Ca_M_kg = keras.Input(shape=(1,))
Fe_kMol = keras.Input(shape=(1,))
Si_Ratio = keras.Input(shape=(1,))
MoO3_Ratio = keras.Input(shape=(1,))
MoO2_Ratio = keras.Input(shape=(1,))
Ca_Ratio = keras.Input(shape=(1,))
Fe3O4_Ratio = keras.Input(shape=(1,))
Mg_Ratio = keras.Input(shape=(1,))
Fe_Ratio = keras.Input(shape=(1,))
Al_Ratio = keras.Input(shape=(1,))

all_inputs = [
    Target_Al,
    Target_Lime,
    Target_Fluorspar,
    Pred_Heat,
    Mix_Weight,
    RMC_Weight,
    Mo_lbs,
    Mo_lbs_with_DCE_and_Remelt,
    Mo_percent,
    MoO3,
    MoO2,
    S_percent_RMC,
    Ca_percent_RMC,
    Na_percent_RMC,
    Total_AcIn,
    Si_kMol,
    Mo_kMol,
    MoO3_kMol,
    MoO2_kMol,
    Fe3O4_kMol,
    Al_kMol,
    Mg_kMol,
    Ca_m_kMol,
    Ca_f_kMol,
    Base_Lime,
    Ca_RMC_kg,
    Ca_DCE,
    Ca_M_kg,
    Fe_kMol,
    Si_Ratio,
    MoO3_Ratio,
    MoO2_Ratio,
    Ca_Ratio,
    Fe3O4_Ratio,
    Mg_Ratio,
    Fe_Ratio,
    Al_Ratio
]

# Numerical features encoded
Target_Al_encoded = DataPrep.encode_numerical_feature(Target_Al, "TargetAl", train_ds)
Target_Lime_encoded = DataPrep.encode_numerical_feature(Target_Lime, "TargetLime", train_ds)
Target_Fluorspar_encoded = DataPrep.encode_numerical_feature(Target_Fluorspar, "TargetFluorspar", train_ds)
Pred_Heat_encoded = DataPrep.encode_numerical_feature(Pred_Heat, "PredHeat", train_ds)
Mix_Weight_encoded = DataPrep.encode_numerical_feature(Mix_Weight, "MixWeight", train_ds)
RMC_Weight_encoded = DataPrep.encode_numerical_feature(RMC_Weight, "RMCWeight", train_ds)
Mo_lbs_encoded = DataPrep.encode_numerical_feature(Mo_lbs, "Molbs", train_ds)
Mo_lbs_with_DCE_and_Remelt_encoded = DataPrep.encode_numerical_feature(Mo_lbs_with_DCE_and_Remelt, "MolbswithDCEandRemelt", train_ds)
Mo_percent_encoded = DataPrep.encode_numerical_feature(Mo_percent, "Mo%", train_ds)
MoO3_encoded = DataPrep.encode_numerical_feature(MoO3, "MoO3", train_ds)
MoO2_encoded = DataPrep.encode_numerical_feature(MoO2, "MoO2", train_ds)
S_percent_RMC_encoded = DataPrep.encode_numerical_feature(S_percent_RMC, "S%RMC", train_ds)
Ca_percent_RMC_encoded = DataPrep.encode_numerical_feature(Ca_percent_RMC, "Ca%RMC", train_ds)
Na_percent_RMC_encoded = DataPrep.encode_numerical_feature(Na_percent_RMC, "Na%RMC", train_ds)
Total_AcIn_encoded = DataPrep.encode_numerical_feature(Total_AcIn, "TotalAcIn", train_ds)
Si_kMol_encoded = DataPrep.encode_numerical_feature(Si_kMol, "SikMol", train_ds)
Mo_kMol_encoded = DataPrep.encode_numerical_feature(Mo_kMol, "MokMol", train_ds)
MoO3_kMol_encoded = DataPrep.encode_numerical_feature(MoO3_kMol, "MoO3kMol", train_ds)
MoO2_kMol_encoded = DataPrep.encode_numerical_feature(MoO2_kMol, "MoO2kMol", train_ds)
Fe3O4_kMol_encoded = DataPrep.encode_numerical_feature(Fe3O4_kMol, "Fe3O4kMol", train_ds)
Al_kMol_encoded = DataPrep.encode_numerical_feature(Al_kMol, "AlkMol", train_ds)
Mg_kMol_encoded = DataPrep.encode_numerical_feature(Mg_kMol, "MgkMol", train_ds)
Ca_m_kMol_encoded = DataPrep.encode_numerical_feature(Ca_m_kMol, "Ca(m)kMol", train_ds)
Ca_f_kMol_encoded = DataPrep.encode_numerical_feature(Ca_f_kMol, "Ca(f)kMol", train_ds)
Base_Lime_encoded = DataPrep.encode_numerical_feature(Base_Lime, "BaseLime", train_ds)
Ca_RMC_kg_encoded = DataPrep.encode_numerical_feature(Ca_RMC_kg, "CaRMCkg", train_ds)
Ca_DCE_encoded = DataPrep.encode_numerical_feature(Ca_DCE, "CaDCE", train_ds)
Ca_M_kg_encoded = DataPrep.encode_numerical_feature(Ca_M_kg, "Ca(M)kg", train_ds)
Fe_kMol_encoded = DataPrep.encode_numerical_feature(Fe_kMol, "FekMol", train_ds)
Si_Ratio_encoded = DataPrep.encode_numerical_feature(Si_Ratio, "SiRatio", train_ds)
MoO3_Ratio_encoded = DataPrep.encode_numerical_feature(MoO3_Ratio, "MoO3Ratio", train_ds)
MoO2_Ratio_encoded = DataPrep.encode_numerical_feature(MoO2_Ratio, "MoO2Ratio", train_ds)
Ca_Ratio_encoded = DataPrep.encode_numerical_feature(Ca_Ratio, "CaRatio", train_ds)
Fe3O4_Ratio_encoded = DataPrep.encode_numerical_feature(Fe3O4_Ratio, "Fe3O4Ratio", train_ds)
Mg_Ratio_encoded = DataPrep.encode_numerical_feature(Mg_Ratio, "MgRatio", train_ds)
Fe_Ratio_encoded = DataPrep.encode_numerical_feature(Fe_Ratio, "FeRatio", train_ds)
Al_Ratio_encoded = DataPrep.encode_numerical_feature(Al_Ratio, "AlRatio", train_ds)

all_features = layers.concatenate(
    [
        Target_Al_encoded,
        Target_Lime_encoded,
        Target_Fluorspar_encoded,
        Pred_Heat_encoded,
        Mix_Weight_encoded,
        RMC_Weight_encoded,
        Mo_lbs_encoded,
        Mo_lbs_with_DCE_and_Remelt_encoded,
        Mo_percent_encoded,
        MoO3_encoded,
        MoO2_encoded,
        S_percent_RMC_encoded,
        Ca_percent_RMC_encoded,
        Na_percent_RMC_encoded,
        Total_AcIn_encoded,
        Si_kMol_encoded,
        Mo_kMol_encoded,
        MoO3_kMol_encoded,
        MoO2_kMol_encoded,
        Fe3O4_kMol_encoded,
        Al_kMol_encoded,
        Mg_kMol_encoded,
        Ca_m_kMol_encoded,
        Ca_f_kMol_encoded,
        Base_Lime_encoded,
        Ca_RMC_kg_encoded,
        Ca_DCE_encoded,
        Ca_M_kg_encoded,
        Fe_kMol_encoded,
        Si_Ratio_encoded,
        MoO3_Ratio_encoded,
        MoO2_Ratio_encoded,
        Ca_Ratio_encoded,
        Fe3O4_Ratio_encoded,
        Mg_Ratio_encoded,
        Fe_Ratio_encoded,
        Al_Ratio_encoded
    ]
)

x = layers.Dropout(0.5)(all_features)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)

output = layers.Dense(27, activation='softmax')(x)

model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

model.fit(train_ds, epochs=10, validation_data=val_ds)
#model.save('C:/Users/matth/Desktop/MTD-NN')


sample = {
    "TargetAl": 180,
    "TargetLime": 83,
    "TargetFluorspar": 120,
    "PredHeat": 454.545204715413,
    "MixWeight": 7336,
    "RMCWeight": 3876,
    "Molbs": 5192.61739668,
    "MolbswithDCEandRemelt": 0,
    "Mo%": 60.767693498452,
    "MoO3": 82.1222464375826,
    "MoO2": 8.04251033652324,
    "S%RMC": 0.0521775025799794,
    "Ca%RMC": 0.345770381836945,
    "Na%RMC": 0.0826971104231166,
    "TotalAcIn": 265.299127572089,
    "SikMol": 35.8789035243859,
    "MokMol": 24.5503001876173,
    "MoO3kMol": 22.1137854100368,
    "MoO2kMol": 2.43651477758044,
    "Fe3O4kMol": 3.1501044464339,
    "AlkMol": 6.92320978502595,
    "MgkMol": 0.64710407239819,
    "Ca(m)kMol": 0.35750249500998,
    "Ca(f)kMol": 2.03836725171019,
    "BaseLime": 77.2789675848303,
    "CaRMCkg": 13.40206,
    "CaDCE": 27.5,
    "Ca(M)kg": 14.3287,
    "FekMol": 12.1638173679499,
    "SiRatio": 13.7382551799346,
    "MoO3Ratio": 43.3895620490826,
    "MoO2Ratio": 4.24928708620012,
    "CaRatio": 1.68383941451124,
    "Fe3O4Ratio": 9.94283921171987,
    "MgRatio": 0.214437022900763,
    "FeRatio": 9.2604852780807,
    "AlRatio": 2.54618593238822
}


input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)
predictions = [ '%.0f' % (elem*100) for elem in predictions[0]]
predictions = [ elem + '%' for elem in predictions]
d = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','Z']
res = dict(zip(d, predictions))

print(
    "MTD-NN predicts this mix will have the following burn notes: {i} ".format(i=res)
)
