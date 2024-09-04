from pyresparser import ResumeParser

data = ResumeParser(
    "sharmaji_resume.pdf"
).get_extracted_data()


print(data)
