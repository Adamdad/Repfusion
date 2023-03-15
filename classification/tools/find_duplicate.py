import pandas as pd
import os
root = "/Users/xingyiyang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/7a3671d9cc49f8fbe0d260c990887f29/Message/MessageTemp/60cdc86db4ee39ae13d0a0c6c0c19385/File/"
ee2026 = pd.read_excel(os.path.join(root, "ee2026 NameList_130223.xlsx"))
# print(ee2026.head())
ee2211 = pd.read_csv(os.path.join(root, "2023-02-14T1417_Grades-EE2211.csv"))
# print(ee2211.head())
both_df = pd.read_excel(os.path.join(root, "副本Copy of EE2211 n EE2026 emrolm_students taking both modules.xlsx"))
both_df_gongfan = pd.read_excel(os.path.join(root, "results_ID.xlsx"))


ee2026_student = ee2026["Student ID"].values
ee2211_student = ee2211['Integration ID'].values
# print(ee2026_student)
# print(ee2211_student)

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

both_student_myresults = intersection(ee2211_student, ee2026_student)

print(both_student_myresults)
print("Num students {}".format(len(both_student_myresults)))

both_df_student = both_df["Student ID"].values
print(both_df_student)
print("Num students {}".format(len(both_df_student)))

both_df_gongfan = both_df_gongfan["EE2211"].values[:105]
print(both_df_gongfan)
print("Num students {}".format(len(both_df_gongfan)))

assert set(both_student_myresults) == set(both_df_student) == set(both_df_gongfan)