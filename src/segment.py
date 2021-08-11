'''
segment.py

This file has function to segment the avec data into
2-5 second segments that will be used for training,
validation, and testing
'''

import os
import pandas as pd



# this function gets all the ~2-5 second responses from the transcripts
def get_segments(transcript_path):
    start_times = []
    end_times = []
    responses = pd.read_csv(transcript_path)
    last_start = 0
    for index, row in responses.iterrows():
        start = row["Start_Time"]
        end = row["End_Time"]

        if start < last_start:
            continue

        # get the length of a given response
        segment_length = float(end)-float(start)

        # a valid segment should be between 2-5 seconds
        if  segment_length > 2.3 and segment_length < 5.1:  
            start_times.append(start)
            end_times.append(end)
        # a response that is longer can be split into smaller segments
        elif segment_length > 5.1:
            curr = start
            # while there is 5 seconds left in the response, get the next 5 seconds
            while curr + 5.0 < end:
                start_times.append(curr)
                end_times.append(curr+5.0)
                curr += 5.0
            # if the last bit is longer than ~2 seconds, then add it
            if end-curr > 2.3:
                start_times.append(curr)
                end_times.append(end)
        last_start = start
        
    return start_times,end_times

# this function writes the segments to the labels csv
def write_segment_csv(patient_id,start_times, end_times, label, csv_path):
    if os.path.exists(csv_path):
        pass
    else:
        f = open(csv_path,"x")
        f.write("patient_id,start,stop,PHQ_Moving_Score")
        f.close()

    labels_frame = pd.read_csv(csv_path)
    i=len(labels_frame)
    for start,end in zip(start_times,end_times):
        labels_frame.loc[i] = [patient_id,start,end,label]
        i+=1

    labels_frame.to_csv(csv_path,index=False)



if __name__ == "__main__":
    # first load the train,val,test splits
    train_split = pd.read_csv("../../avec_data/train_split.csv")
    val_split = pd.read_csv("../../avec_data/dev_split.csv")
    test_split = pd.read_csv("../../avec_data/test_split.csv")

    labels_frame = pd.read_csv("../../avec_data/Detailed_PHQ8_Labels.csv",index_col="Participant_ID")

    # print("train split...")
    # for index, row in train_split.iterrows():
    #     id = row["Participant_ID"]
    #     print(id)
    #     starts,ends = get_segments("../../avec_data/"+str(id)+"_P/"+str(id)+"_Transcript.csv")
    #     if int(id) in labels_frame.index:
    #         score = labels_frame.at[int(id),"PHQ_8Moving"]
    #         write_segment_csv(id,starts,ends,score,"//totoro/perception-working/Geffen/avec_data/train_metadata.csv")

    # print("val split...")
    # for index, row in val_split.iterrows():
    #     id = row["Participant_ID"]
    #     print(id)
    #     starts,ends = get_segments("../../avec_data/"+str(id)+"_P/"+str(id)+"_Transcript.csv")
    #     if int(id) in labels_frame.index:
    #         score = labels_frame.at[int(id),"PHQ_8Moving"]
    #         write_segment_csv(id,starts,ends,score,"//totoro/perception-working/Geffen/avec_data/val_metadata.csv")

    print("test split...")
    for index, row in test_split.iterrows():
        id = row["Participant_ID"]
        print(id)
        starts,ends = get_segments("../../avec_data/"+str(id)+"_P/"+str(id)+"_Transcript.csv")
        if int(id) in labels_frame.index:
            score = labels_frame.at[int(id),"PHQ_8Moving"]
            write_segment_csv(id,starts,ends,score,"//totoro/perception-working/Geffen/avec_data/test_metadata.csv")