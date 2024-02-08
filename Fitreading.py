import os
import gzip
import shutil

# Define the folder directory containing the GZIP-compressed FIT files
main_folder = r'C:\Users\ezran\Downloads\export_32489751\activities'

# Specify the directory where you want to extract the FIT files
extract_dir = r'D:\fit files'

# Create the extract directory if it doesn't exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Iterate through all files in the main folder
for root, dirs, files in os.walk(main_folder):
    for file_name in files:
        # Check if the file is a GZIP-compressed FIT file
        if file_name.endswith('.fit.gz'):
            gz_file_path = os.path.join(root, file_name)

            # Create a corresponding FIT file name by removing the '.gz' extension
            fit_file_name = os.path.splitext(file_name)[0] + '.fit'
            fit_file_path = os.path.join(extract_dir, fit_file_name)

            # Open and decompress the GZIP file, and then write it to the FIT file
            with gzip.open(gz_file_path, 'rb') as gz_file, open(fit_file_path, 'wb') as fit_file:
                shutil.copyfileobj(gz_file, fit_file)

            print(f"Extracted: {fit_file_name} to {extract_dir}")


#
# from fitparse import FitFile
#
# def read_fit_file(file_path):
#     fitfile = FitFile(file_path)
#
#     for record in fitfile.get_messages():
#         print(f"Message: {record.name}")
#
#         for field in record.fields:
#             print(f"  Field: {field.name} - Value: {field.value}")
#
#
# file_path = r"C:\Users\ezran\Downloads\fit files\fit files\Sexy_people_go_crazy_too_you_know_Read_a_people_magazine_.fit"# Replace with the path to your FIT file
# read_fit_file(file_path)
