from datetime import date
import json
from Activity import Activity 

#for handling with all the activities/json file data
class Activity_Manager:
    
    def read_data(self):
        try:
            with open("User_logs.json", "r") as file:
                #loads the data from the file
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            # file missing or empty on first run, start fresh
            return []
    def __init__(self ):
        #priv attr-data received from the json file 
        self.__data=self.read_data()
           
    def get_data(self):
        return self.__data
        
    
    def add_activity(self, category, start_time,end_time, notes) :
        today=str(date.today().strftime("%m-%d-%Y"))
        # create activity dict
        activity=Activity(category,start_time, end_time, notes ).input_to_dict()
        activity["date"]=today
        self.__data.append(activity)
        
         

                
            
       
                
    #add the data to the file/creates new and adds if non exist
    def write_data(self):
        with open("User_logs.json", "w") as file:
            #dump-write the python data into the json file, indent-5 key-value pairs
            json.dump(self.__data, file, indent=5)     
            