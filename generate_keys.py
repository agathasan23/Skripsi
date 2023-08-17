import streamlit as st
import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["Daffa Pratama", "Novelia Agatha Santoso", "Nurul Fadillah"]
usernames = ["daffa.pratama", "novelia.santoso", "nurul.fadillah"]
passwords = ["XXX", "XXX", "XXX"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent/"hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)