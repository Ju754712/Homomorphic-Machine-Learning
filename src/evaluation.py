import pandas as pd


data = pd.read_csv('src/csv/autoencoder_more.csv')  

print(data["encoder_input_encryption_time"].sum()/101)
print(data["encoder_plain_time"].sum()/101)
print(data["encoder_more_time"].sum()/101)
print(data["encoder_output_decryption_time"].sum()/101)

print(data["decoder_input_encryption_time"].sum()/101)
print(data["decoder_plain_time"].sum()/101)
print(data["decoder_more_time"].sum()/101)
print(data["decoder_output_decryption_time"].sum()/101)