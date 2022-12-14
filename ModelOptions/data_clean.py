#system requirements
    # Last updated: 2022-12-13T20:29:07.454649-05:00

    # Python implementation: CPython
    # Python version       : 3.7.13
    # IPython version      : 7.31.1

    # Compiler    : MSC v.1916 64 bit (AMD64)
    # OS          : Windows
    # Release     : 10
    # Machine     : AMD64
    # Processor   : AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
    # CPU cores   : 16
    # Architecture: 64bit
    
    # numpy     : 1.21.5
    # pandas    : 1.3.5
    # sys       : 3.7.13 (default, Mar 28 2022, 08:03:21) [MSC v.1916 64 bit (AMD64)]

#dicionary used to cleanup data
cleaner = {'addr_state': {'PA': 0.0,  'SD': 1.0,  'IL': 2.0,  'NJ': 3.0,
  'GA': 4.0,  'MN': 5.0,  'SC': 6.0,  'RI': 7.0,  'TX': 8.0,  'NC': 9.0,  'CA': 10.0,  'VA': 11.0,
  'AZ': 12.0,  'NY': 13.0,  'IN': 14.0,  'MD': 15.0,  'KS': 16.0,  'NM': 17.0,  'AL': 18.0,  'WA': 19.0,
  'MO': 20.0,  'OH': 21.0,  'LA': 22.0,  'FL': 23.0,  'CO': 24.0,  'MI': 25.0,  'TN': 26.0,
  'DC': 27.0,  'MA': 28.0,  'WI': 29.0,  'HI': 30.0,  'VT': 31.0,  'DE': 32.0,  'NH': 33.0,  'NE': 34.0,
  'CT': 35.0,  'OR': 36.0,  'AR': 37.0,  'MT': 38.0,  'NV': 39.0,  'WV': 40.0,  'WY': 41.0,  'OK': 42.0,
  'KY': 43.0,  'MS': 44.0,  'ME': 45.0,  'UT': 46.0,  'ND': 47.0,  'AK': 48.0,  'ID': 50.0,  'IA': 51.0},
 'application_type': {'Individual': 0.0, 'Joint App': 1.0},
 'debt_settlement_flag': {'N': 0.0, 'Y': 1.0},
 'disbursement_method': {'Cash': 1.0, 'DirectPay': 2.0},
 'emp_length': {'10+ years': 10.0,  '3 years': 3.0,  '4 years': 4.0,
  '6 years': 6.0,  '1 year': 1.0,  '7 years': 7.0,  '8 years': 8.0,
  '5 years': 5.0,  '2 years': 2.0,  '9 years': 9.0,  '< 1 year': 1.0},
 'grade': {'C': 0.0,  'B': 1.0,  'F': 2.0,  'A': 3.0,  'E': 4.0,  'D': 5.0,  'G': 6.0},
 'hardship_flag': {'N': 0.0, 'Y': 1.0},
 'home_ownership': {'MORTGAGE': 0.0,  'RENT': 1.0,
      'OWN': 2.0,  'ANY': 3.0,'NONE': 4.0,  'OTHER': 5.0},
 'initial_list_status': {'w': 0.0, 'f': 1.0},
 'purpose': {'debt_consolidation': 0.0,  'small_business': 1.0,
      'home_improvement': 2.0,  'major_purchase': 3.0,
      'credit_card': 4.0,  'other': 5.0,
      'house': 6.0,  'vacation': 7.0,
      'car': 8.0,  'medical': 9.0,
      'moving': 10.0,  'renewable_energy': 11.0,
      'wedding': 12.0,  'educational': 13.0,},
 'pymnt_plan': {'n': 0.0, 'y': 1.0},
 'sub_grade': {'C4': 0.0,  'C1': 1.0,
      'B4': 2.0,  'C5': 3.0,
      'F1': 4.0,  'C3': 5.0,
      'B2': 6.0,  'B1': 7.0,
      'A2': 8.0,  'B5': 9.0,
      'C2': 10.0,  'E2': 11.0,
      'A4': 12.0,  'E3': 13.0,
      'A1': 14.0,  'D4': 15.0,
      'F3': 16.0,  'D1': 17.0,
      'B3': 18.0,  'E4': 19.0,
      'D3': 20.0,  'D2': 21.0,
      'D5': 22.0,  'A5': 23.0,
      'F2': 24.0,  'E1': 25.0,
      'F5': 26.0,  'E5': 27.0,
      'A3': 28.0,  'G2': 29.0,
      'G1': 30.0,  'G3': 31.0,
      'G4': 32.0,  'F4': 33.0,
      'G5': 34.0},
 'term': {' 36 months': 0.0, ' 60 months': 1.0},
 'verification_status': {'Not Verified': 0.0,
  'Source Verified': 1.0,
  'Verified': 2.0}}



def data_clean:
     
    mypath = "../../LG_Resources/Resources/lending-club/accepted_2007_to_2018Q4.csv/accepted_2007_to_2018Q4.csv"
    # inputmypath = input("Type in the relative path to the csv file with the loan database:")
    
    #read data into df
    df = pd.read_csv(
        Path(mypath),  
        infer_datetime_format=True,
        parse_dates = True,
        low_memory=False)

    #remove unnecessary coulmns
    nan_values = pd.DataFrame(df.isna().sum(),columns = ["NAN Count"]).reset_index()
    drop_columns = nan_values[(nan_values['NAN Count'] > 500000)]['index'].tolist() 
    drop_columns.extend(['id','url','title','zip_code']) #deemed unnecessary after review 
    df = df.drop(drop_columns, axis=1)

    #setup values for good/bad loans using loan_status values
    status = df['loan_status'].dropna().unique().tolist()
    defaultstatus = status[2:-2]
    defaultstatus.append(status[-1])
    goodstatus = [i for i in status if i not in defaultstatus]

    df['Default'] = np.where(df['loan_status'].isin(defaultstatus), 1, 0)

    #change types of categorical attibutes to categorical attributes
    df = df.replace(cleaner)
    df = modeldf.fillna(0.00)

    return df 

def scale_df(df):
    #List of categorical columns
    categorical = list(cleaner.keys())
    noncategorical = [i for i in df.columns if i not in categorical]    

    #scale data
    scaled_data = StandardScaler().fit_transform(modeldf[noncategorical])

    #create new  df
    df_scaled = pd.DataFrame(scaled_data) #add scaled data
    df_scaled.columns = noncategorical #rename columns
    df_scaled[categorical] = modeldf[categorical] #add back categorical data (does not need scaling)

    return df_scaled
