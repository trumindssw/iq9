echo " ----------> Step 1 - Installing the Requirements ---------- "
pip3 install -r requirements.txt

echo " ----------> Step 2 - Pre-processing the packets - Combining all processed packets into a single DataFrame ---------- "
echo " ----------> Generating Feather file ---------- "
python3 nndpi_preprocessing.py \
  --n_jobs=1 \
  --pcap_dir=./CompletePCAPs/ \
  --processed_pcap_dir=./ProcessedPackets/ \
  --max_len=1500 \
  --one_df=False

echo " ----------> Step 3 - inference - human-readable source code into a lower-level language called 'bytecode'"
python3 inference-pc.py

echo " ----------> Step 4 - converting feather files to numpy format which can run on IQ9"
python3 python3 feather-file-to-numpy.py