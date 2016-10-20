#!/bin/bash

#this will add the upstream branch (the main / team one)
git remote add upstream https://github.com/epfml/ML_course.git
git remote -v

#this will synchronize user's branch with the team branch (on your computer only)
git fetch upstream
git checkout master
git merge upstream/master

#to push the synchronized branch to your github
# fix conflicts (if any): git mergetool
# git commit -am "merged with team branch"
# git push

