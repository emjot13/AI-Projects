#!/bin/bash
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo -e "script takes 3 obligatory arguments
1. max number of results
2. hashtag to search for
3. path to file to save results to
and one optional 
4. date after the results are saved"
  return 0
fi

if [ $# -ge 4 ]
then
    snscrape --since $4 --max-results $1 twitter-hashtag $2 > $3
else
    snscrape --max-results $1 twitter-hashtag $2 > $3
fi