class Activity: #data container
    
    def __init__(self, category, start_time, end_time, notes):
        self.category=category
        self.start_time=start_time
        self.end_time=end_time
        self.notes=notes
       
        
   
    #converts instance to dict
    def input_to_dict(self):  
        return {            
        "category":self.category,
        "start_time":self.start_time,
        "end_time":self.end_time,
        "notes": self.notes
        }
        