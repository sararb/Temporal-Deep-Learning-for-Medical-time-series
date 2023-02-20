import pandas as pd
import pickle
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_cv_raw_data(data_path,
                    out_dir,
                    min_measurements,
                    train_size=0.8):
    """
    :param out_dir:
    :param train_size: the size of training sample, the remaining samples will be equally split into valid and test data
    :param data_path: the path to the raw data-frame where each line represent one patient
    :param min_measurements: The minimum values a patient should have to be included in the study
    :return: Save pickles files related to train, valid and test data and return status "ok"
    """
    data = pd.read_csv(data_path, sep = ';', encoding = 'latin-1')
    data = data.dropna(subset = ['seq_hba1c_sans_doublons'])
    data.seq_hba1c_sans_doublons = data.seq_hba1c_sans_doublons.apply(lambda x: [float(e) for e in x.split('|') if x])
    data.seq_timedelta_sans_doublons = data.seq_timedelta_sans_doublons.apply(
        lambda x: [float(e) for e in x.split('|') if x])
    general_retino_info = data[data.seq_hba1c_sans_doublons.apply(lambda x: len(x) >= min_measurements)]
    # encode target column
    general_retino_info.statut = general_retino_info.statut.astype(int)
    print("nan values found: %s" %general_retino_info['age_decouverte'].isnull().sum())
    general_retino_info['age_decouverte'] = general_retino_info['age_decouverte'].fillna(0)
    print(general_retino_info['age_decouverte'].isnull().sum())
    X_train, X_test, y_train, y_test = train_test_split(
        general_retino_info,
        general_retino_info.statut.astype(int),
        shuffle = True,
        train_size = train_size,
        random_state = 42,
        stratify = general_retino_info.statut.astype(int))
    # Normalize hba1c data
    std = MinMaxScaler()
    possible_hba1c = X_train['seq_hba1c_sans_doublons'].values
    possible_hba1c = np.array([hbac for e in possible_hba1c for hbac in e])
    std.fit(possible_hba1c.reshape(-1, 1))
    X_train['seq_hba1c_norm'] = X_train['seq_hba1c_sans_doublons'].apply(lambda x:
                                                                         std.transform(
                                                                             np.array(x).reshape(-1, 1)).reshape(-1))
    X_test['seq_hba1c_norm'] = X_test['seq_hba1c_sans_doublons'].apply(lambda x:
                                                                       std.transform(
                                                                           np.array(x).reshape(-1, 1)).reshape(-1))

    # Normalize timedelta data
    std_time = MinMaxScaler()
    possible_time = X_train['seq_timedelta_sans_doublons'].values
    possible_time = np.array([hba1c for e in possible_time for hba1c in e])
    std_time.fit(possible_time.reshape(-1, 1))
    X_train['timedelta_norm'] = X_train['seq_timedelta_sans_doublons'].apply(lambda x:
                                                                             std_time.transform(
                                                                                 np.array(x).reshape(-1, 1)).reshape(
                                                                                 -1))
    X_test['timedelta_norm'] = X_test['seq_timedelta_sans_doublons'].apply(lambda x:
                                                                           std_time.transform(
                                                                               np.array(x).reshape(-1, 1)).reshape(
                                                                               -1))
    # normalize side_tensor non suivi
    duree = MinMaxScaler()
    duree.fit(X_train['duree_non_suivi'].values.reshape(-1, 1))
    X_train['duree_non_suivi_norm'] = duree.transform(X_train['duree_non_suivi'].values.reshape(-1, 1)).reshape(-1)
    X_test['duree_non_suivi_norm'] = duree.transform(X_test['duree_non_suivi'].values.reshape(-1, 1)).reshape(-1)

    # normalize age at discovery
    age = MinMaxScaler()
    age.fit(X_train['age_decouverte'].values.reshape(-1, 1))
    X_train['age_decouverte_norm'] = age.transform(X_train['age_decouverte'].values.reshape(-1, 1)).reshape(-1)
    X_test['age_decouverte_norm'] = age.transform(X_test['age_decouverte'].values.reshape(-1, 1)).reshape(-1)

    X_test, X_valid, y_test, y_valid = train_test_split(X_test,
                                                        y_test,
                                                        shuffle = True,
                                                        train_size = 0.5,
                                                        random_state = 42)
    os.makedirs(os.path.join(out_dir, 'dictionaries'), exist_ok = True)
    # Save training and test data
    for set_, X in zip(['Train', 'Valid', 'Test'], [X_train, X_valid, X_test]):
        X["seq_length"] = X.seq_timedelta_sans_doublons.apply(lambda x: len(x))
        info_patient = X.set_index('IdPatient').to_dict('index')
        [d.update({'patient_id': key}) for key, d in info_patient.items()]
        # save the set data to disk
        filename = '%s/dictionaries/%d_patients_min_%s_seq_infos_%s.sav' % (out_dir, len(X),
                                                                            min_measurements, set_)
        print("save %s data into %s" % (set_, filename))
        pickle.dump(list(info_patient.values()), open(filename, 'wb'))
    return 'ok'


def main():
    data_path = '/home/rabhi/dataset/temporal_hba1c/modelisation_hba1c_retino_v3/modelisation_seq_hba1c_retino_mpatient_T1D_v3.csv'
    for mes in [3, 5, 10, 15]:
        get_cv_raw_data(data_path = data_path,
                        out_dir = "/home/rabhi/dataset/temporal_hba1c/modelisation_hba1c_retino_v3",
                        min_measurements = mes)


if __name__ == "__main__":
    main()
