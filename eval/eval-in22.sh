SYN='BI-en-gu'
TASK='IN22-Conv'
SRC='eng_Latn'
TGT='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 
SYN='BI-en-gu'
TASK='IN22-Gen'
SRC='eng_Latn'
TGT='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 
SYN='BI-en-gu'
TASK='IN22-Conv'
TGT='eng_Latn'
SRC='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 
SYN='BI-en-gu'
TASK='IN22-Gen'
TGT='eng_Latn'
SRC='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 

SYN='BI-en-gu_syn-parallel'
TASK='IN22-Conv'
SRC='eng_Latn'
TGT='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 
SYN='BI-en-gu_syn-parallel'
TASK='IN22-Gen'
SRC='eng_Latn'
TGT='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 
SYN='BI-en-gu_syn-parallel'
TASK='IN22-Conv'
TGT='eng_Latn'
SRC='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 
SYN='BI-en-gu_syn-parallel'
TASK='IN22-Gen'
TGT='eng_Latn'
SRC='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 

SYN='BI-en-gu_syn-diffsrc'
TASK='IN22-Conv'
SRC='eng_Latn'
TGT='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 
SYN='BI-en-gu_syn-diffsrc'
TASK='IN22-Gen'
SRC='eng_Latn'
TGT='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 
SYN='BI-en-gu_syn-diffsrc'
TASK='IN22-Conv'
TGT='eng_Latn'
SRC='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 
SYN='BI-en-gu_syn-diffsrc'
TASK='IN22-Gen'
TGT='eng_Latn'
SRC='guj_Gujr'
SIZE='tiny'
python3 eval-in22.py $SYN $TASK $SRC $TGT $SIZE 

