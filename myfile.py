import pickle
import streamlit as st #pip install streamlit 
from streamlit_option_menu import option_menu #option menu streamlit extras
from streamlit_extras.add_vertical_space import add_vertical_space #adding space vertically
from sklearn.metrics import (confusion_matrix,accuracy_score,precision_score,recall_score,f1_score)
from sklearn.ensemble import RandomForestClassifier #random forest algorithm
from sklearn.model_selection import train_test_split #split data train and data test
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import plotly.express as px
import pandas as pd #pip install pandas openpyxL

# -- BACKGROUND DESIGN -- #
st.set_page_config(page_title="Website Prediksi Karyawan",layout="wide")
page_bg_img = """
        <style> 
        [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1588345921523-c2dcdb7f1dcd?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80");
        background-size: cover;
        }

        [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
        }
        </style>
        """
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- HEADER --- #
row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((2, 0.2, 1, 0.1))
row0_1.title("Project Prediksi Karyawan Resign")

with row0_2: add_vertical_space()
row0_2.subheader("Download Template CSV")
text_contents = '''
NIK, NAME, GENDER, AGE, EDUCATION_LEVEL_ID, MARITAL_STATUS, DEPARTMENT, PERSONAL_SUB_AREA, EMPLOYEE_STATUS, EMPLOYEE_LEVEL, DAILY_HOUR, YEARS_IN_COMPANY, DISTANCE, DURATION, LABEL
'''
row0_2.download_button('Download Template', text_contents, 'Template Prediksi.csv', 'text/csv')

# -- FUNGSI CONVERT DAILY HOUR MENJADI SECONDS -- #
def to_seconds(s):
    hr, minutes, sec = [float(x) for x in s.split(':')]
    return hr*3600 + minutes*60 + sec

# -- FUNGSI DATA CLEANING -- #
def dropna(datasets):
    datasets = datasets.dropna(axis=0)
    return datasets

def clean_data_automated(datasets):
    datasets = datasets.dropna(axis=0)
    datasets.Daily_Hour = datasets.Daily_Hour.apply(to_seconds)
    datasets.Daily_Hour = datasets.Daily_Hour / 3600
    datasets.Distance = datasets.Distance / 1000
    datasets.Duration = datasets.Duration / 60
    datasets.Age = datasets.Age.astype('int64')
    datasets.Years_in_Company = datasets.Years_in_Company.astype('int64')
    datasets_clean = datasets.drop(columns = ['NIK','Name','Personal_Sub_Area'])
    datasets_clean.Marital_Status = datasets_clean.Marital_Status.map({'Lajang':1, 'Menikah':2, 'Cerai Mati':3, 'Cerai Hidup':3, 'Duda/Janda':4})
    datasets_clean.Employee_Level = datasets_clean.Employee_Level.map({'Level 1':1, 'Level 2**':2, 'Level 2*':3, 'Level 3**':4, 'Level 3*': 5, 'Level 4**':6, 'Level 4*':7, 'Level 5**': 8, 'Level 5*':9, 'Level 6': 10, 'Level 7':11})
    datasets_clean.Employee_Status = datasets_clean.Employee_Status.map({'Magang':1, 'Kontrak pertama':2, 'Kontrak Perpanjangan':3, 'Kontrak pembaharuan':3, 'Tetap':4})
    datasets_clean.Gender = datasets_clean.Gender.map({'Laki-laki':1, 'Perempuan':2})
    datasets_clean.Education_Level_Id = datasets_clean.Education_Level_Id.astype('int64')
    datasets_clean.Departemen = datasets_clean.Departemen.map({'Executive':1,'FIRA':2,'Cost Control': 3, 'Testing Laboratory':4, 'Procurement':5, 'Administration':6,'Research and Development': 7, 'Finance': 8,'Commercial':9,'Production':10})
    datasets_clean.Label = datasets_clean.Label.dropna(axis=0)
    datasets_clean.Label = datasets_clean.Label.map({'actives':0,'terminates':1})
    return datasets_clean
            
# -- NAVIGATION BAR -- #
selected = option_menu(menu_title=None, options=["Visualisasi","Update Model","Prediksi"], icons=["bar-chart-line-fill","arrow-up-circle-fill","diagram-3-fill"], default_index=0, orientation="horizontal", styles={
                "icon": {"color": "black", "font-size": "25px"},
                "nav-link": {"--hover-color": "#BD9797"},
                "nav-link-selected": {"background-color": "#960018",},
            }
        )

# -- VISUALIZATION PAGE -- #    
if selected == "Visualisasi":
    # -- FUNGSI UPLOAD DATA -- #
    data_file=st.file_uploader("Upload file karyawan yang ingin divisualisasikan", type=["csv"])

    if data_file is None:
        st.info("Upload File Dataset Dulu Kaka🙂", icon="🚩")

    if data_file is not None:
        df = pd.read_csv(data_file)
        df = dropna(df)
        data_clean = clean_data_automated(df)

        # -- KOLOM 1 2 HISTOGRAM -- #
        kol1, kol2=st.columns(2)

        with kol1:
            fig_gender = px.histogram(df, x="Gender", color="Label", barmode="group", height=400)
            st.subheader("Histogram Gender")
            st.plotly_chart(fig_gender, theme=None, use_container_width=True)
            
        with kol2:
            fig_age = px.histogram(df, x="Age", color="Label", barmode="group", height=400)
            st.subheader("Histogram Umur")
            st.plotly_chart(fig_age, theme=None, use_container_width=True)

        with kol1:
            fig_departemen = px.sunburst(df, path=['Label','Departemen'], height=400)
            st.subheader("Sunburst Departemen")
            st.plotly_chart(fig_departemen, theme=None, use_container_width=True)

        with kol2:
            fig_marital = px.sunburst(df, path=['Label','Marital_Status'], height=400)
            st.subheader("Sunburst Marital")
            st.plotly_chart(fig_marital, theme=None, use_container_width=True)

        # -- DATAFRAME DATASET -- #
        st.subheader("Dataframe Dari Dataset")
        st.dataframe(df)

        # -- DATAFRAME DATA CLEANING -- #
        st.subheader("Dataframe Dari Data Cleaning")
        st.dataframe(data_clean, use_container_width=True) 
        
        #use_container_width = true itu apa?

        # -- CONVERT DATA TO ARRAY -- #
        X=data_clean.drop(columns=['Label'], axis=1) #Menentuan Features, ':' berarti memilih semua baris, dan ':-1' mengabaikan kolom terakhir
        y=data_clean['Label'] #Menentukan Label, ':' berarti memilih semua baris, dan '-1:' mengabaikan semua kolom kecuali kolom terakhir
        
        # -- SET DATA TRAIN AND DATA TEST -- #
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=42) #Mengambil 75% untuk Data Training 

        #kenapa random_state=42 ?
        
        # -- CLASSIFIER -- #
        clf=RandomForestClassifier(n_estimators=1000) #Membuat model
        clf.fit(X_train,y_train) #Training model yang dibuat
        ypredict=clf.predict(X)
        
        # -- ACCURACY & CONFUSION MATRIX & PRECISION & RECALL & F1-SCORE -- #
        kol1, kol2, kol3, kol4, kol5 = st.columns(5, gap="small")

        with kol1 :
            st.subheader("Confusion Matrix")
            st.text(confusion_matrix(y,ypredict))
            
        with kol2 :
            st.subheader("Accuracy")
            st.text(accuracy_score(y,ypredict)*100)
            
        with kol3 :
            st.subheader("Precision")
            st.text(precision_score(y,ypredict))
        
        with kol4 : 
            st.subheader("Recall")
            st.text(recall_score(y,ypredict))
        
        with kol5 :
            st.subheader("F1-Score")
            st.text(f1_score(y,ypredict))

# -- UPDATE MODEL PAGE -- #
if selected == "Update Model":
    # -- PENJELASAN -- #
    st.info("WARNING : Perlu diketahui bahwa fitur update model ini akan mempengaruhi hasil prediksi. Pastikan anda mengupload file dataset karyawan terupdate untuk memperbarui model prediksi.", icon="🚩")
    
    # -- INPUT DATASET UNTUK MODELING BARU -- #
    data_baru = st.file_uploader("Upload file baru untuk update model algoritma", type=["csv"])

    if data_baru is None:
        st.info("Upload File Dataset Yang Baru Kaka🙂", icon="🚩")

    if data_baru is not None:
        db = pd.read_csv(data_baru)
        db = dropna(db)
        db_clean = clean_data_automated(db)

        st.subheader("Dataframe File Update")
        st.dataframe(db)

        # -- CONVERT DATA TO ARRAY -- #
        X=db_clean.drop(columns=['Label'], axis=1) #Menentuan Features, ':' berarti memilih semua baris, dan ':-1' mengabaikan kolom terakhir
        y=db_clean['Label'] #Menentukan Label, ':' berarti memilih semua baris, dan '-1:' mengabaikan semua kolom kecuali kolom terakhir

        # -- SET DATA TRAIN AND DATA TEST -- #
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=42) #Mengambil 75% untuk Data Training

        # -- CLASSIFIER -- #
        modelrf=RandomForestClassifier(n_estimators=1000) #Membuat model
        modelrf.fit(X_train,y_train) #Training model yang dibuat
        ypredict=modelrf.predict(X)

        # -- SAVE UPDATE MODEL -- #
        if st.button('Update Model'):
            filename = 'modelrf_final.sav'
            pickle.dump(modelrf, open(filename,"wb"))
            st.balloons()
            st.success("Model berhasil diupdate🎉")

# -- PREDICTION PAGE -- #
if selected == "Prediksi":
    percent = "%"
    # -- CLEANING INPUT DATA -- #
    def clean_data_csv(datasets):
        datasets.Daily_Hour = datasets.Daily_Hour.apply(to_seconds)
        datasets.Daily_Hour = datasets.Daily_Hour / 3600
        datasets.Distance = datasets.Distance / 1000
        datasets.Duration = datasets.Duration / 60
        datasets.Age = datasets.Age.astype('int64')
        datasets.Years_in_Company = datasets.Years_in_Company.astype('int64')
        newdata_clean = datasets.drop(columns = ['NIK','Name','Personal_Sub_Area','Label'],axis=1)
        newdata_clean.Marital_Status = newdata_clean.Marital_Status.map({'Lajang':1, 'Menikah':2, 'Cerai Mati':3, 'Cerai Hidup':3, 'Duda/Janda':4})
        newdata_clean.Employee_Level = newdata_clean.Employee_Level.map({'Level 1':1, 'Level 2**':2, 'Level 2*':3, 'Level 3**':4, 'Level 3*': 5, 'Level 4**':6, 'Level 4*':7, 'Level 5**': 8, 'Level 5*':9, 'Level 6': 10, 'Level 7':11})
        newdata_clean.Employee_Status = newdata_clean.Employee_Status.map({'Magang':1, 'Kontrak pertama':2, 'Kontrak Perpanjangan':3, 'Kontrak pembaharuan':3, 'Tetap':4})
        newdata_clean.Gender = newdata_clean.Gender.map({'Laki-laki':1, 'Perempuan':2})
        newdata_clean.Education_Level_Id = newdata_clean.Education_Level_Id.map({'SD':1, 'SMP':2, 'SMA':3, 'SMK':4, 'Diploma':5, 'S1':6, 'S2':8, 'Post/Grad Diploma':9,'S3':10})
        newdata_clean.Departemen = newdata_clean.Departemen.map({'Executive':1,'FIRA':2,'Cost Control': 3, 'Testing Laboratory':4, 'Procurement':5, 'Administration':6,'Research and Development': 7, 'Finance': 8,'Commercial':9,'Production':10})
        return newdata_clean
    
    # -- FUNGSI CLEANING DATA INPUT NON CSV -- #
    def clean_data_input(dataframe):
        dataframe = dataframe
        dataframe.Marital_Status = dataframe.Marital_Status.map({'Lajang':1, 'Menikah':2, 'Cerai Mati':3, 'Cerai Hidup':3, 'Duda/Janda':4})
        dataframe.Employee_Level = dataframe.Employee_Level.map({'Level 1':1, 'Level 2**':2, 'Level 2*':3, 'Level 3**':4, 'Level 3*': 5, 'Level 4**':6, 'Level 4*':7, 'Level 5**': 8, 'Level 5*':9, 'Level 6': 10, 'Level 7':11})
        dataframe.Employee_Status = dataframe.Employee_Status.map({'Magang':1, 'Kontrak Pertama':2, 'Kontrak Perpanjangan':3, 'Kontrak Pembaharuan':3, 'Tetap':4})
        dataframe.Gender = dataframe.Gender.map({'Laki-laki':1, 'Perempuan':2})
        dataframe.Education_Level_Id = dataframe.Education_Level_Id.map({'SD':1, 'SMP':2, 'SMA':3, 'SMK':4, 'Diploma':5, 'S1':6, 'S2':8, 'Post/Grad Diploma':9,'S3':10})
        dataframe.Departemen = dataframe.Departemen.map({'Executive':1,'FIRA':2,'Cost Control': 3, 'Testing Laboratory':4, 'Procurement':5, 'Administration':6,'Research and Development': 7, 'Finance': 8,'Commercial':9,'Production':10})
        return dataframe

    # -- FUNGSI DROP LABEL DAN NIK DARI INPUT CSV -- #
    def drop_label(dataset):
        dataset = dataset.drop(columns = ['Label','NIK'],axis=1)
        return dataset
                                
    with st.expander("Input CSV"):
        # -- INPUT USER NEW TESTING DATA BY CSV -- #
        st.subheader("Input file CSV")
        newfile = st.file_uploader("Upload file CSV untuk prediksi data baru")

        if newfile is not None:
            file = pd.read_csv(newfile)
            file = dropna(file)

            # -- PENGHAPUSAN KOLOM NIK & LABEL -- #
            file_without_label = drop_label(file)
            file_clean = clean_data_csv(file)

            # -- LOAD MODEL LAST UPDATE -- #
            load_model = pickle.load(open("modelrf_final.sav","rb"))                   
            ypredict = (load_model.predict_proba(file_clean))

            # -- RESET INDEX UNTUK BISA CONCATENATE -- #
            file_without_label = file_without_label.reset_index()
            file_without_label = file_without_label.drop(columns=['index'])
            file_clean = pd.DataFrame(file_clean)

            # -- MENGUBAH DATA PREDIKSI MENJADI STRING + % -- #
            final = list()
            for item in ypredict:
                num = item[1]*100
                predict = str(round(num,5)) + percent
                final.append(predict)
            
            # -- MENGUBAH LIST PREDIKSI MENJADI DATAFRAME DENGAN LABEL 'PREDIKSI' -- #
            datahasil = pd.DataFrame(final, columns=['Prediksi'])
            
            st.subheader("Dataframe Dari Hasil Prediksi")
            st.dataframe(file_clean, use_container_width=True)

            # -- MENGGABUNGKAN DATAFRAME FILE CSV DENGAN HASIL PREDIIKSI -- #
            newpredict = pd.concat([file_without_label, datahasil], axis=1)
            st.dataframe(newpredict)

            # -- CONVERT DATA TO CSV -- #
            def convert_dat(dat):
                return dat.to_csv(index=False).encode('utf-8')

            hasil = convert_dat(newpredict)
            st.download_button("Download Hasil Prediksi", hasil, file_name="Hasil_Prediksi.csv", mime='text/csv')

    with st.expander("Input Manual"):
        # -- INPUT USER NEW TESTING DATA -- #                
        st.subheader("Input value yang ingin diprediksi")
        with st.form('Form 1'):
                
            # -- FORM INPUT -- #
            age = st.number_input('Age', min_value= 20, max_value=80, step=1)
            gender = st.selectbox('Gender', ('Laki-laki', 'Perempuan'))
            departemen = st.selectbox('Department', ('Administration','Commercial','Cost Center','Executive','Finance','FIRA','Procurement','Production','Research and Development','Testing Laboratory'))
            marital_status = st.selectbox('Marital Status', ('Lajang','Menikah','Cerai Hidup','Cerai Mati','Duda/Janda'))
            educational_level_id = st.selectbox('Pendidikan Terakhir', ('SD','SMP','SMA','SMK','Diploma','S1','S2','Post/Grad Diploma','S3'))
            employee_status = st.selectbox('Status Karyawan', ('Magang','Kontrak Pertama','Kontrak Perpanjangan','Kontrak Pembaharuan','Tetap'))
            employee_level = st.selectbox('Level Karyawan', ('Level 1','Level 2**','Level 2*','Level 3**','Level 3*','Level 4**','Level 4*','Level 5**','Level 5*','Level 6','Level 7'))
            daily_hour = st.time_input('Daily Hours')
            years_in_company = st.number_input('Years in Company', min_value= 0, max_value=50, step=1)
            distance = st.number_input('Jarak (KM)', min_value= 0.1, max_value=100.0, step=1.0)
            duration = st.time_input('Durasi (HH/MM)')
            
            # -- CONVERTS FUNCTION-- #
            def to_seconds_input(s):
                hr, minutes = [float(x) for x in s.split(':')]
                return hr*3600 + minutes*60

            def time_to_string(t):
                t = t.strftime("%H:%M")
                return t

            # -- CONVERTING DATA -- #
            daily_hour = time_to_string(daily_hour)
            duration = time_to_string(duration)

            daily_hour = to_seconds_input(daily_hour)
            duration = to_seconds_input(duration)

            # -- MEMBUAT DICTIONARY DARI INPUT DATA -- #
            datalist = {                
                "Gender": [gender],
                "Age": [age],
                "Education_Level_Id" : [educational_level_id],
                "Marital_Status" : [marital_status],                
                "Departemen": [departemen],
                "Employee_Status": [employee_status],
                "Employee_Level": [employee_level],                                
                "Daily_Hour": [daily_hour],
                "Years_in_Company": [years_in_company],                
                "Distance": [distance],
                "Duration": [duration]
            }
            # -- MENGUBAH DATALIST DICTIONARY MENJADI DATAFRAME -- #
            dataframe = pd.DataFrame(data=datalist)

            # -- TOMBOL SUBMIT INPUT USER -- #
            submitted = st.form_submit_button('Submit')
            if submitted is True :
                st.success('Submit Berhasil')   
                
                # -- MENAMPILKAN DATA CLEANING DARI INPUT USER -- #
                newdata = clean_data_input(dataframe)
                
                # -- MEMPREDIKSI INPUTAN USER -- #
                load_model = pickle.load(open("modelrf_final.sav","rb"))                   
                hasil_new = load_model.predict_proba(newdata)
                pre = round(hasil_new[0][1],5)
                                                        
                st.subheader('Prediksi Karyawan Tersebut Resign Adalah')
                st.text(hasil_new)
                result=str(pre*100) + percent 

                # -- DESKRIPSI CARA BACA ANGKA PREDIKSI -- #
                st.caption("Pada angka disamping menunjukkan nilai berapa peluang karyawan tersebut mendekati kejadian 0 atau 1. Sehingga bisa disimpulkan bahwa persentase kemungkinan karyawan tersebut untuk resign sebesar")
                st.subheader(result)
