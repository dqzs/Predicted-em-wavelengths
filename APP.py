import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from autogluon.tabular import TabularPredictor

# 加载Autogluon模型
model = TabularPredictor.load("ag-20240829_082340")

# Streamlit界面
st.title("荧光发射波长预测器")

# 获取用户输入的SMILES格式
smiles_input = st.text_input("请输入SMILES格式的化学结构:")

# 检查输入是否为空
if smiles_input:
    # 使用RDKit读取SMILES格式
    mol = Chem.MolFromSmiles(smiles_input)
    
    # 检查SMILES是否有效
    if mol is None:
        st.error("输入的SMILES格式无效，请检查后重新输入。")
    else:
        # 计算分子描述符
        descs = Descriptors.descList
        molecule_descriptors = [descs.index(desc) for desc in descs if desc != 'MolWt' and desc != 'HeavyAtomMolWt']
        molecule_features = [Descriptors.MolWt(mol), Descriptors.HeavyAtomMolWt(mol)] + [Descriptors._descList[desc](mol) for desc in molecule_descriptors]
        
        # 构建特征DataFrame
        features_df = pd.DataFrame([molecule_features])
        
        # 使用Autogluon模型进行预测
        predicted_wavelength = model.predict(features_df)[0]
        
        # 显示预测结果
        st.write(f"预测的荧光发射波长为: {predicted_wavelength:.2f} nm")
else:
    st.write("请在上方输入SMILES格式的化学结构。")