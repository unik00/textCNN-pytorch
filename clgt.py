import subprocess
scorer = 'data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl'
temporary_file_path = 'data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/temporary_file.txt'
test_file_key_path = 'data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/TEST_FILE_KEY.TXT'
print(subprocess.check_output(["perl", scorer, temporary_file_path, test_file_key_path]
                              # stderr=subprocess.STDOUT,
                              # shell=True
      )
)
