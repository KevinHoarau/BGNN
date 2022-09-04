from BML.data import Dataset
from BML import utils
from BML.transform import DatasetTransformation

#################
# Data collection

folder = "data/"
dataset = Dataset(folder)

dataset.setParams({
    "PrimingPeriod": 10*60, # 10 hours of priming data
    "IpVersion": [4], # only IPv4 routes
    "Collectors": ["rrc04","rrc05"], 
    "UseRibsPriming": True
})

dataset.setPeriodsOfInterests([
    {
        "name": "TTNet",
        "label": "anomaly",
        "start_time": utils.getTimestamp(2004, 12, 24, 9, 20, 0) - 60*30, 
        "end_time": utils.getTimestamp(2004, 12, 24, 9, 20, 0) + 60*30, 
    },
    {
        "name": "IndoSat",
        "label": "anomaly",
        "start_time": utils.getTimestamp(2014, 4, 2, 18, 25, 0) - 60*30, 
        "end_time": utils.getTimestamp(2014, 4, 2, 18, 25, 0) + 60*30, 
    },
    {
        "name": "TM",
        "label": "anomaly",
        "start_time": utils.getTimestamp(2015, 6, 12, 8, 43, 0) - 60*30, 
        "end_time": utils.getTimestamp(2015, 6, 12, 8, 43, 0) + 60*30
    },
    {
        "name": "AWS",
        "label": "anomaly",
        "start_time": utils.getTimestamp(2016, 4, 22, 17, 10, 0) - 60*30, 
        "end_time": utils.getTimestamp(2016, 4, 22, 17, 10, 0) + 60*30
    },
    {
        "name": "Google",
        "label": "anomaly",
        "start_time": utils.getTimestamp(2017, 8, 25, 3, 22, 0) - 60*30, 
        "end_time": utils.getTimestamp(2017, 8, 25, 3, 22, 0) + 60*30, 
    },
    {
        "name": "ChinaTelecom",
        "label": "anomaly",
        "start_time": utils.getTimestamp(2019, 6, 6, 9, 44, 0) - 60*30, 
        "end_time": utils.getTimestamp(2019, 6, 6, 9, 44, 0) + 60*30, 
    },
    {
        "name": "India",
        "label": "anomaly",
        "start_time": utils.getTimestamp(2021, 4, 16, 13, 48, 0) - 60*30, 
        "end_time": utils.getTimestamp(2021, 4, 16, 13, 48, 0) + 60*30, 
    },
        {
        "name": "TTNet",
        "label": "no_anomaly",
        "start_time": utils.getTimestamp(2004, 12, 24, 9, 20, 0) - 60*30 - 24*3600, 
        "end_time": utils.getTimestamp(2004, 12, 24, 9, 20, 0) + 60*30 - 24*3600, 
    },
    {
        "name": "IndoSat",
        "label": "no_anomaly",
        "start_time": utils.getTimestamp(2014, 4, 2, 18, 25, 0) - 60*30 - 24*3600, 
        "end_time": utils.getTimestamp(2014, 4, 2, 18, 25, 0) + 60*30 - 24*3600, 
    },
    {
        "name": "TM",
        "label": "no_anomaly",
        "start_time": utils.getTimestamp(2015, 6, 12, 8, 43, 0) - 60*30 - 24*3600, 
        "end_time": utils.getTimestamp(2015, 6, 12, 8, 43, 0) + 60*30 - 24*3600
    },
    {
        "name": "AWS",
        "label": "no_anomaly",
        "start_time": utils.getTimestamp(2016, 4, 22, 17, 10, 0) - 60*30 - 24*3600, 
        "end_time": utils.getTimestamp(2016, 4, 22, 17, 10, 0) + 60*30 - 24*3600
    },
    {
        "name": "Google",
        "label": "no_anomaly",
        "start_time": utils.getTimestamp(2017, 8, 25, 3, 22, 0) - 60*30 - 24*3600, 
        "end_time": utils.getTimestamp(2017, 8, 25, 3, 22, 0) + 60*30 - 24*3600, 
    },
    {
        "name": "ChinaTelecom",
        "label": "no_anomaly",
        "start_time": utils.getTimestamp(2019, 6, 6, 9, 44, 0) - 60*30 - 24*3600, 
        "end_time": utils.getTimestamp(2019, 6, 6, 9, 44, 0) + 60*30 - 24*3600, 
    },
    {
        "name": "India",
        "label": "no_anomaly",
        "start_time": utils.getTimestamp(2021, 4, 16, 13, 48, 0) - 60*30 - 24*3600, 
        "end_time": utils.getTimestamp(2021, 4, 16, 13, 48, 0) + 60*30 - 24*3600, 
    },
])

# run the data collection
utils.runJobs(dataset.getJobs(), folder+"collect_jobs", nbProcess=16) 

# features extraction every 2 minute
datTran = DatasetTransformation(folder, "BML.transform", "Graph")

datTran.setParams({
        "global":{
            "Name": "WeightedGraph_2",
            "Period": 2,
            "relabel_nodes": True,
            "weighted": True,
            "NbSnapshots": 30,
            "SkipIfExist": True,
        }
    })

# run the data transformation
utils.runJobs(datTran.getJobs(), folder+"transform_jobs") 