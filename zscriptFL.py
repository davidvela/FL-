#Import dependencies
from subprocess import call


with open("README.md", "a") as f:
     f.write("new line\n")

#Commit Message
commit_message = "Adding sample files"

#Stage the file 
call('git add .', shell = True)

# Add your commit
call('git commit -m "'+ "test commit message" +'"', shell = True)

#Push the new or update files
call('git push origin master', shell = True)