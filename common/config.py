
algo_config = [
    {
        "algorithmNum": 1,
        "modelName": "car",
        "modelChar": "车辆人员识别",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有车辆、人员、电动车，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/wurenji_new.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "汽车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "person",
                "paramType": "double",
                "paramChar": "人员",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "person",
                "paramType": "double",
                "paramChar": "电动车",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 2,
        "modelName": "rubbish",
        "modelChar": "垃圾识别",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有建筑垃圾、生活垃圾，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/live_build_rubbish.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "build_rubbish",
                "paramType": "double",
                "paramChar": "建筑垃圾",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "live_rubbish",
                "paramType": "double",
                "paramChar": "生活垃圾",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 4,
        "modelName": "Smoke",
        "modelChar": "烟雾识别",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有烟雾，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "SAM3",
        "text_prompt": "smoke",
        "modelPath": "/tensorrt/checkpoints/sam3.pt",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "smoke",
                "paramType": "double",
                "paramChar": "烟雾",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 17,
        "modelName": "river_floats",
        "modelChar": "河道漂浮物检测",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有水藻、河道漂浮物，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/riverv8.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "algae",
                "paramType": "double",
                "paramChar": "水藻",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "float",
                "paramType": "double",
                "paramChar": "漂浮物",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },

    {
        "algorithmNum": 18,
        "modelName": "helmet_vest",
        "modelChar": "安全帽反光衣检测",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label可能是安全帽、反光衣，请你帮我复审一下是否存在安全帽、反光衣，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/helmet_vest.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "helmet",
                "paramType": "double",
                "paramChar": "安全帽",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "vest",
                "paramType": "double",
                "paramChar": "反光衣",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },

    {
        "algorithmNum": 21,
        "modelName": "gongdi",
        "modelChar": "工地巡检",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label可能是裸土、安全帽、高空作业车、挖掘机、卡车、木材、钢材、水泥车、积水，请你帮我复审一下是否存在裸土、安全帽、高空作业车、挖掘机、卡车、木材、钢材、水泥车、积水，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/gongdi.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "soil",
                "paramType": "double",
                "paramChar": "裸土",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "hat",
                "paramType": "double",
                "paramChar": "安全帽",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "skywork",
                "paramType": "double",
                "paramChar": "高空作业车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "digger",
                "paramType": "double",
                "paramChar": "挖掘机",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "truck",
                "paramType": "double",
                "paramChar": "卡车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "wood",
                "paramType": "double",
                "paramChar": "木材",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "steel",
                "paramType": "double",
                "paramChar": "钢材",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "mixed",
                "paramType": "double",
                "paramChar": "水泥车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "water",
                "paramType": "double",
                "paramChar": "积水",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },

    {
        "algorithmNum": 22,
        "modelName": "peoplepeople",
        "modelChar": "人员检测",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label是人员，请你帮我复审一下是否为人员，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/people.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "people",
                "paramType": "double",
                "paramChar": "人员",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },

    {
        "algorithmNum": 23,
        "modelName": "poppy",
        "modelChar": "罂粟识别",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有罂粟，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/poppy.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "poppy",
                "paramType": "double",
                "paramChar": "罂粟",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 24,
        "modelName": "Peeling paint on the wall",
        "modelChar": "墙皮脱落",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有墙皮脱落，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "SAM3",
        "text_prompt": "Peeling paint on the wall",
        "modelPath": "/tensorrt/checkpoints/sam3.pt",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "wall",
                "paramType": "double",
                "paramChar": "墙皮脱落",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 25,
        "modelName": "street_stall",
        "modelChar": "占道经营",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有流动摊贩、占道经营，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/street_stall.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "vendor",
                "paramType": "double",
                "paramChar": "流动摊贩",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "street_stall",
                "paramType": "double",
                "paramChar": "占道经营",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 28,
        "modelName": "gongdi_cars",
        "modelChar": "不同器械车辆",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有翻斗车、挖掘机、高空作业车、叉车、压土车、水泥车，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/gongdi_cars.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "truck",
                "paramType": "double",
                "paramChar": "翻斗车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "digger",
                "paramType": "double",
                "paramChar": "挖掘机",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "skywork",
                "paramType": "double",
                "paramChar": "高空作业车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "folklift",
                "paramType": "double",
                "paramChar": "叉车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "roller",
                "paramType": "double",
                "paramChar": "压土车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "mixed",
                "paramType": "double",
                "paramChar": "水泥车",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 29,
        "modelName": "hat",
        "modelChar": "安全帽",
        "model_prompt": "描述一下图片的内容,并检查画面中人员是否佩戴安全帽，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/hat.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "hat",
                "paramType": "double",
                "paramChar": "安全帽",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 30,
        "modelName": "steel_wood",
        "modelChar": "钢材木材",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有钢材、木材，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/steel_wood.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "steel",
                "paramType": "double",
                "paramChar": "钢材",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "wood",
                "paramType": "double",
                "paramChar": "木材",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 31,
        "modelName": "soil_water",
        "modelChar": "裸土积水",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有裸土、积水，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/soil_water.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "soil",
                "paramType": "double",
                "paramChar": "裸土",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "water",
                "paramType": "double",
                "paramChar": "积水",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 32,
        "modelName": "garbage",
        "modelChar": "环卫垃圾检测",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有垃圾袋、纸盒、其他垃圾、易拉罐、塑料瓶，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/garbage.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "garbage",
                "paramType": "double",
                "paramChar": "垃圾袋",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "cardbox",
                "paramType": "double",
                "paramChar": "纸盒",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "waste",
                "paramType": "double",
                "paramChar": "其他垃圾",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "canna",
                "paramType": "double",
                "paramChar": "易拉罐",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "bottle",
                "paramType": "double",
                "paramChar": "塑料瓶",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 33,
        "modelName": "banner",
        "modelChar": "横幅检测",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有横幅，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/banner.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "banner",
                "paramType": "double",
                "paramChar": "横幅",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 34,
        "modelName": "road",
        "modelChar": "道路裂缝",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有道路裂缝、修补、井盖、网状缺陷，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/road.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "crack",
                "paramType": "double",
                "paramChar": "裂缝",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "patch",
                "paramType": "double",
                "paramChar": "修补",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "manhole",
                "paramType": "double",
                "paramChar": "井盖",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "net",
                "paramType": "double",
                "paramChar": "网状",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 35,
        "modelName": "lprnet",
        "modelChar": "车辆违停（车牌抓拍）",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有车辆车牌并进行识别，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/license.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "license",
                "paramType": "double",
                "paramChar": "车牌",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 37,
        "modelName": "ship",
        "modelChar": "船只检测",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有船只，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/ship.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "ship",
                "paramType": "double",
                "paramChar": "船",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 38,
        "modelName": "illegal",
        "modelChar": "违规搭棚",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label是违规搭棚，请你帮我复审一下是否为违规搭棚，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/illegal.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "illegal",
                "paramType": "double",
                "paramChar": "违规搭棚",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 39,
        "modelName": "sunroom",
        "modelChar": "阳光房",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有阳光房，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/sunroom.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "sunroom",
                "paramType": "double",
                "paramChar": "阳光房",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 40,
        "modelName": "vehicle",
        "modelChar": "电动车",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有电动车，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/vehicle.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "vehicle",
                "paramType": "double",
                "paramChar": "电动车",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 41,
        "modelName": "fishing",
        "modelChar": "垂钓",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label是垂钓，请你帮我复审一下是否为垂钓，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/fishing.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "fishing",
                "paramType": "double",
                "paramChar": "垂钓",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 42,
        "modelName": "cone",
        "modelChar": "路障",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有锥桶、柱桶类路障，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/cone.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "cone",
                "paramType": "double",
                "paramChar": "锥桶",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "barrel",
                "paramType": "double",
                "paramChar": "柱桶",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 43,
        "modelName": "onebox",
        "modelChar": "一号快递箱",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有一号快递箱，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/onebox.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "onebox",
                "paramType": "double",
                "paramChar": "一号快递箱",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 44,
        "modelName": "parking",
        "modelChar": "车位车辆",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有停车位、停放的车辆，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/parking.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "tcw",
                "paramType": "double",
                "paramChar": "停车位",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "车",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 45,
        "modelName": "ddc_wtk",
        "modelChar": "骑车安全",
        "model_prompt": "描述一下图片的内容,并检查骑电动车人员是否佩戴头盔，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/ddc_wtk.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "tk",
                "paramType": "double",
                "paramChar": "头盔",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "ddc",
                "paramType": "double",
                "paramChar": "电动车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "wtk",
                "paramType": "double",
                "paramChar": "无头盔",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 46,
        "modelName": "stone",
        "modelChar": "石块识别",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有石块，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/stone.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "stone",
                "paramType": "double",
                "paramChar": "石块",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 47,
        "modelName": "GF",
        "modelChar": "光伏板",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有车辆、非机动车、人员、光伏板及光伏板瑕疵，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/GF.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "Car",
                "paramType": "double",
                "paramChar": "车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "Non-motorized vehicles",
                "paramType": "double",
                "paramChar": "非机动车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "person",
                "paramType": "double",
                "paramChar": "人",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "photovoltaic panels",
                "paramType": "double",
                "paramChar": "光伏板",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "single spote",
                "paramType": "double",
                "paramChar": "单个斑点",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "multie spotes",
                "paramType": "double",
                "paramChar": "多个斑点",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "streak flaw",
                "paramType": "double",
                "paramChar": "条纹瑕疵",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "component_damaged",
                "paramType": "double",
                "paramChar": "部件损坏",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 48,
        "modelName": "face",
        "modelChar": "人脸识别",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有人脸，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/face.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "face",
                "paramType": "double",
                "paramChar": "人脸",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 52,
        "modelName": "smoke_head",
        "modelChar": "吸烟检测",
        "model_prompt": "描述一下图片的内容,并检查画面中人员是否有吸烟行为，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/smoke_head.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "head",
                "paramType": "double",
                "paramChar": "头",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "smokeing",
                "paramType": "double",
                "paramChar": "吸烟",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 54,
        "modelName": "illegal_parking",
        "modelChar": "机动车违停",
        "model_prompt": "描述一下图片的内容,并检查画面中机动车是否存在违停情况，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/car_tcw-obb.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "车",
                "defValue": 0.8,
                "valueScope": ""
            },
            {
                "paramName": "tcw",
                "paramType": "double",
                "paramChar": "停车位",
                "defValue": 0.6,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 55,
        "modelName": "fire_exit",
        "modelChar": "消防通道占用",
        "model_prompt": "描述一下图片的内容,并检查画面中消防通道是否被占用，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/xiaofangtongdao.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "车",
                "defValue": 0.4,
                "valueScope": ""
            },
            {
                "paramName": "fire_exit",
                "paramType": "double",
                "paramChar": "消防通道",
                "defValue": 0.4,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 56,
        "modelName": "cleaner",
        "modelChar": "无人清扫车",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有无人清扫车，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/cleaner.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "cleaner",
                "paramType": "double",
                "paramChar": "无人清扫车",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 57,
        "modelName": "vehicle_illegal",
        "modelChar": "非机动车违停",
        "model_prompt": "描述一下图片的内容,并检查画面中非机动车是否存在违停情况，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/electric_vehicle_tcw-obb.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "electric_vehicle",
                "paramType": "double",
                "paramChar": "非机动车",
                "defValue": 0.4,
                "valueScope": ""
            },
            {
                "paramName": "tcw",
                "paramType": "double",
                "paramChar": "停车位",
                "defValue": 0.4,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 58,
        "modelName": "Electric_tower",
        "modelChar": "电线塔异物检测",
        "model_prompt": "描述一下图片的内容,并检查电线塔上是否有鸟巢、风筝、气球、火焰、人员、猴子等异物，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/Electric_tower.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "Nest",
                "paramType": "double",
                "paramChar": "鸟巢",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "Kite",
                "paramType": "double",
                "paramChar": "风筝",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "Balloon",
                "paramType": "double",
                "paramChar": "气球",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "Fire",
                "paramType": "double",
                "paramChar": "火焰",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "Person",
                "paramType": "double",
                "paramChar": "人",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "Monkey",
                "paramType": "double",
                "paramChar": "猴子",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 64,
        "modelName": "tree",
        "modelChar": "枯木检测",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有枯木，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/tree_new.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "Dead_wood",
                "paramType": "double",
                "paramChar": "枯木",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 65,
        "modelName": "billboard",
        "modelChar": "广告牌检测",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有独立式广告牌、挂式广告牌、交通标志，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/billboard.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "Freestanding_billboard",
                "paramType": "double",
                "paramChar": "独立式广告牌",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "Mounted_billboard",
                "paramType": "double",
                "paramChar": "挂式广告牌",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "Road_sign",
                "paramType": "double",
                "paramChar": "交通标志",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 66,
        "modelName": "waste",
        "modelChar": "废弃物检测",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有废弃物，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/rubbish.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "Rubbish",
                "paramType": "double",
                "paramChar": "废弃物",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 67,
        "modelName": "flowerpot",
        "modelChar": "花盆识别",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有花盆，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/huapen.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "flowerpot",
                "paramType": "double",
                "paramChar": "花盆",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 68,
        "modelName": "trash",
        "modelChar": "工地垃圾检测",
        "model_prompt": "描述一下图片的内容,并检查工地画面中是否有垃圾堆放，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/trash.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "trash",
                "paramType": "double",
                "paramChar": "垃圾",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 69,
        "modelName": "car_speed",
        "modelChar": "车辆测速",
        "model_prompt": "描述一下图片的内容,并识别车道、检测车辆并进行测速，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        # "modelPath": {'69':'/tensorrt/models/gs_lanes_seg.engine','999':'/tensorrt/models/gs_car_v2.engine'},
        "modelPath": '/tensorrt/models/gs_lanes_seg.engine',
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "left_lane",
                "paramType": "double",
                "paramChar": "左车道",
                "defValue": 0.8,
                "valueScope": ""
            },
            {
                "paramName": "right_lane",
                "paramType": "double",
                "paramChar": "右车道",
                "defValue": 0.8,
                "valueScope": ""
            },
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "车辆",
                "defValue": 0.7,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 70,
        "modelName": "trash",
        "modelChar": "垃圾堆放",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label是小区垃圾，请你帮我复审一下是否为小区垃圾，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/jinfeng_xiaoqu_laji.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "trash",
                "paramType": "double",
                "paramChar": "小区垃圾",
                "defValue": 0.4,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 71,
        "modelName": "manhole_cover",
        "modelChar": "井盖检测",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有井盖及异常，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/jinfeng_jinggai.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "manhole_cover",
                "paramType": "double",
                "paramChar": "井盖",
                "defValue": 0.6,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 72,
        "modelName": "soil",
        "modelChar": "裸土识别",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label是裸土，请你帮我复审一下是否为裸土，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/jinfeng_luotu.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "soil",
                "paramType": "double",
                "paramChar": "裸土",
                "defValue": 0.6,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 73,
        "modelName": "puddle",
        "modelChar": "水坑识别",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有积水水坑，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/jinfeng_shuikeng.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "puddle",
                "paramType": "double",
                "paramChar": "积水",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 74,
        "modelName": "arable_land",
        "modelChar": "土地耕种",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label是耕地，请你帮我复审一下是否为耕地，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/jinfeng_gengdi.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "arable_land",
                "paramType": "double",
                "paramChar": "耕地",
                "defValue": 0.6,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 75,
        "modelName": "gs",
        "modelChar": "高速应急车道", # '/tensorrt/models/gs_car_v2.engine'
        'model_prompt': "这张图片是YOLOv8检测后生成的框，label是应急车道占用，只有车辆完全出现在应急车道线内才算违规，请你帮我复审一下是否为应急车道占用，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": '/tensorrt/models/gs_v2_seg.engine',
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "lane",
                "paramType": "double",
                "paramChar": "应急车道",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 76,
        "modelName": "charging_pile",
        "modelChar": "充电桩检测",
        "model_prompt": "描述一下图片的内容,label分别是充电枪、充电桩、汽车，并检查画面中是否有充电桩、充电枪、汽车，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/cdz_new.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "plug",
                "paramType": "double",
                "paramChar": "充电枪",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "charger",
                "paramType": "double",
                "paramChar": "充电桩",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "汽车",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 77,
        "modelName": "fishing",
        "modelChar": "钓鱼者",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label是钓鱼，请你帮我复审一下是否为钓鱼，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/best_fishing.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "fishing",
                "paramType": "double",
                "paramChar": "钓鱼",
                "defValue": 0.6,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 78,
        "modelName": "plastic-bag",
        "modelChar": "塑料袋",
        "model_prompt": "描述一下图片的内容,并检查画面中是塑料袋的情况，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/best_plastic_bag.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "plastic_bag",
                "paramType": "double",
                "paramChar": "塑料袋",
                "defValue": 0.6,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 79,
        "modelName": "gs",
        "modelChar": "高速车辆检测",
        'model_prompt': "这个是经过YOLOv8模型推理的结果，需要你再次复审一下，有无车祸和应急车道占用的情况，应急车道占用是车辆完全行驶入最右侧车道才判定为违规占用，并说出依据，返回json格式，例子：{gs_text}",
        "modelPath": '/tensorrt/models/gs_car_v2.engine',
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "lane",
                "paramType": "double",
                "paramChar": "应急车道",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "车辆",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": None,
        "modelName": "OnlyCar",
        "modelChar": "汽车",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有小汽车，大货车以及不同种车辆的情况，返回结果以json的格式，返回格式：{类别: True/False}",
        "modelType": "SAM3",
        "text_prompt": "Car",
        "modelPath": "/tensorrt/checkpoints/sam3.pt",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "汽车",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": None,
        "modelName": "OnlyFire",
        "modelChar": "火焰识别",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有火焰和烟雾的情况，返回结果以json的格式，返回格式：{类别: True/False}",
        "modelType": "SAM3",
        "text_prompt": "fire",
        "modelPath": "/tensorrt/checkpoints/sam3.pt",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "fire",
                "paramType": "double",
                "paramChar": "火焰",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 80,
        "modelName": "Carcount",
        "modelChar": "车流量统计",
        # "model_prompt": "描述一下图片的内容,并检查画面中是否有火焰和烟雾的情况，返回结果以json的格式，返回格式：{类别: True/False}",
        # "modelType": "SAM3",
        # "text_prompt": "fire",
        "modelPath": "/tensorrt/models/gs_car_v2.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "汽车",
                "defValue": 0.5,
                "valueScope": ""
            },
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 81,
        "modelName": "Personcount",
        "modelChar": "人流量统计",
        # "model_prompt": "描述一下图片的内容,并检查画面中是否有火焰和烟雾的情况，返回结果以json的格式，返回格式：{类别: True/False}",
        # "modelType": "SAM3",
        # "text_prompt": "fire",
        "modelPath": "/tensorrt/models/people.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "people",
                "paramType": "double",
                "paramChar": "人员",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 83,
        "modelName": "car_tcw_cdz",
        "modelChar": "汽车充电桩停车位检测",
        "model_prompt": "描述一下图片的内容,label分别是充电枪、充电桩、汽车，并检查画面中是否有充电桩、充电枪、汽车，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelPath": "/tensorrt/models/car_tcw_cdz.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "汽车",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "charger",
                "paramType": "double",
                "paramChar": "充电桩",
                "defValue": 0.5,
                "valueScope": ""
            },
            {
                "paramName": "parking",
                "paramType": "double",
                "paramChar": "停车位",
                "defValue": 0.5,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 85,
        "modelName": "zawu",
        "modelChar": "小区垃圾检测",
        "modelPath": "/tensorrt/models/zawu.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "zawu",
                "paramType": "double",
                "paramChar": "垃圾",
                "defValue": 0.7,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 86,
        "modelName": "floating_debris_det_all_new",
        "modelChar": "水面漂浮物检测",
        # "model_prompt": "这张图片是YOLOv11检测后生成的框，label是水面漂浮物(floating_debris)，请你帮我复审一下画面中是否存在水面漂浮废弃物、漂浮杂物或类似垃圾，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/float_debris_det.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "floating_debris",
                "paramType": "double",
                "paramChar": "水面漂浮物",
                "defValue": 0.20,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 87,
        "modelName": "illegal_construction",
        "modelChar": "违规建筑检测",
        "model_prompt": "这张图片是YOLOv11检测后生成的框，label是违规建筑(illegal_construction)或脚手架(scaffold)，请你帮我复审一下画面中是否存在违规建筑或脚手架，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/best.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "illegal_construction",
                "paramType": "double",
                "paramChar": "违规建筑",
                "defValue": 0.40,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 90,
        "modelName": "cyanobacteria",
        "modelChar": "水面蓝藻检测",
        # "model_prompt": "这张图片是YOLOv11检测后生成的框，label是蓝藻(cyanobacteria)，请你帮我复审一下画面中是否存在蓝藻、水华或水面绿色藻类聚集，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/cyanobacteria.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "cyanobacteria",
                "paramType": "double",
                "paramChar": "蓝藻",
                "defValue": 0.30,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 88,
        "modelName": "tcw",
        "modelChar": "停车位检测",
        #"model_prompt": "这张图片是YOLOv11检测后生成的框，label是停车位(tcw)。请你帮我复审一下：只要检测框内存在停车位，就判定为画面中存在停车位；不需要判断该停车位是否真的可用、是否合法或是否有其他属性。返回格式：{'result': True/False, 'according_to': '如果result为True，说明看到停车位的依据；如果result为False，说明未看到停车位的依据'}。返回结果以json形式。",
        "modelType": "segment",
        "modelPath": "/tensorrt/models/tcw.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "tcw",
                "paramType": "double",
                "paramChar": "停车位",
                "defValue": 0.50,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 92,
        "modelName": "jiegan_fire",
        "modelChar": "秸秆焚烧",
        "model_prompt": "描述一下内容，label是灰烬，fire是火焰，smoke是烟雾，并检查画面中是否有灰烬、火焰、烟雾，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/jiegan_fire.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "ashes",
                "paramType": "double",
                "paramChar": "灰烬",
                "defValue": 0.50,
                "valueScope": ""
            },
            {
                "paramName": "fire",
                "paramType": "double",
                "paramChar": "火焰",
                "defValue": 0.50,
                "valueScope": ""
            },
            {
                "paramName": "smoke",
                "paramType": "double",
                "paramChar": "烟雾",
                "defValue": 0.50,
                "valueScope": ""
            },
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 91,
        "modelName": "roadside_booths",
        "modelChar": "占道经营",
        # "model_prompt": "这张图片是YOLOv11检测后生成的框，label是占道经营(roadside_booth)，请你帮我复审一下画面中是否存在占道经营、流动摊贩或占道摊位，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/roadside_booths.engine",
        "withTrack": True,
        "modelParams": [
            {
                "paramName": "roadside_booth",
                "paramType": "double",
                "paramChar": "占道经营",
                "defValue": 0.50,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 93,
        "modelName": "drowning_prevention",
        "modelChar": "防溺水检测",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label是person（人员），但YOLO可能误检。请你只复审检测框内是否存在真实的人：只要检测框内明确存在真实的人，就判定为有溺水风险，不需要判断该人是否真的正在溺水、落水、挣扎或位于水中；但是不要因为label是person就直接判定为True。只有能清楚看到人的头部、躯干、四肢、人体轮廓或人的姿态时，才返回True；如果检测框内不是人，例如土堆、牛、其他动物、石头、树木、杂物、阴影、反光、水面纹理、岸边物体、雕塑、车辆部件，或者目标太小、模糊、遮挡严重、无法确认是人，一律返回False。返回格式:{'result':True or False,'according_to':'如果result为True，说明看到人的依据；如果result为False，说明未看到人的依据'}返回结果以json形式",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/honghu_best.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "person",
                "paramType": "double",
                "paramChar": "有溺水风险",
                "defValue": 0.50,
                "valueScope": ""
            },
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 95,
        "modelName": "car",
        "modelChar": "日常车辆检测",
        #"model_prompt": "这张图片是YOLOv8检测后生成的框，label为'car'或'truck'。请你复审检测框内是否存在真实的车，以判定是否有溺水风险（只要框内确实有人即可，无需判断是否溺水、挣扎或位于水中）。判断时请适度宽松：只要框内内容整体上更像人而非其他物体，即可判定为True。例如，能看出人体轮廓、疑似头部/躯干/四肢的形态、人体姿态、衣物特征、肤色区域，或与周围环境相比明显呈现人体形状，均视为有效依据。即使目标较小、略模糊、部分遮挡，只要仍可合理推断是人，也应返回True。只有在明显不是人的情况下才返回False，例如清晰可见的动物、车辆部件、雕塑、大型杂物、纯阴影或水面纹理，且完全不具备任何人的形态特征。不要因为目标小、模糊或遮挡就轻易否定。返回格式：{'result': True/False, 'according_to': '依据描述（简要说明看到或未看到人的理由）'}，结果以JSON形式输出。",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/car_1280.engine",
        "withTrack": False,
        "imgsz": 1280,
        "modelParams": [
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "汽车",
                "defValue": 0.60,
                "valueScope": ""
            },
            {
                "paramName": "truck",
                "paramType": "double",
                "paramChar": "卡车",
                "defValue": 0.99,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 96,
        "modelName": "smoke",
        "modelChar": "抽烟检测",
        "model_prompt": "这张图片是YOLOv8检测后生成的框，label为'smoke'。请你复审检测框内是否存在真实的抽烟行为（包括手持香烟、烟头明火、烟雾、叼烟等），以判定该区域是否有人在抽烟（只要框内确实有抽烟相关元素即可，无需判断是否正在吸烟或是否违规）。判断时请适度宽松：只要框内内容整体上更像抽烟行为或烟具而非其他物体，即可判定为True。例如，能看出细长条状的香烟、燃烧发红的烟头、飘散的烟雾、手指夹烟的动作、嘴边有烟或滤嘴，或具有典型抽烟姿势的轮廓，均视为有效依据。即使目标较小、略模糊、部分被手部或面部遮挡、光线不足，只要仍可合理推断是抽烟，也应返回True。只有在明显不是抽烟的情况下才返回False，例如清晰的笔状物、手指自然姿态、蒸汽、水雾、无烟动作、或完全不具备任何烟具或吸烟特征。不要因为目标小、模糊或遮挡就轻易否定。返回格式：{'result': True/False, 'according_to': '依据描述（简要说明看到或未看到抽烟行为的理由）'}，结果以JSON形式输出。",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/smoke2.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "person",
                "paramType": "double",
                "paramChar": "人员",
                "defValue": 0.40,
                "valueScope": ""
            },
            {
                "paramName": "smoke",
                "paramType": "double",
                "paramChar": "抽烟",
                "defValue": 0.40,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
{
        "algorithmNum": 106,
        "modelName": "car_person",
        "modelChar": "汽车人员统计",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有车辆、人员，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/car_person_01.engine",
        "withTrack": False,
        "imgsz": 960,
        "modelParams": [
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "车辆",
                "defValue": 0.30,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
        {
        "algorithmNum": 107,
        "modelName": "car",
        "modelChar": "汽车统计",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有车辆，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/car0929.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "car",
                "paramType": "double",
                "paramChar": "车辆",
                "defValue": 0.50,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
{
        "algorithmNum": 108,
        "modelName": "person",
        "modelChar": "人员统计",
        "model_prompt": "描述一下图片的内容,并检查画面中是否有人员，返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "detect",
        "modelPath": "/tensorrt/models/person1010_960.engine",
        "withTrack": False,
        "imgsz": 960,
        "modelParams": [
            {
                "paramName": "person",
                "paramType": "double",
                "paramChar": "人员",
                "defValue": 0.30,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
    {
        "algorithmNum": 111,
        "modelName": "turbid_water",
        "modelChar": "浑浊水体识别",
        "model_prompt": "这张图片是YOLOv11分割后生成的掩膜，label是水体(water)。请你复审该水体区域的水色是否浑浊：浑浊表现为水色偏黄褐、发灰或浑浊不清（悬浮泥沙、污水），清澈表现为青绿色、通透。返回格式:{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}返回结果以json形式",
        "modelType": "segment",
        "modelPath": "/tensorrt/models/turbid_water.engine",
        "withTrack": False,
        "modelParams": [
            {
                "paramName": "water",
                "paramType": "double",
                "paramChar": "水体",
                "defValue": 0.30,
                "valueScope": ""
            }
        ],
        "synStatus": 0
    },
]

TARGET_PROMPT = "这张图片是YOLOv8检测后生成的框"
TARGET_PROMPT_TEMPLATE = (
    "这张图片是YOLOv8检测后生成的框，label是{}，请你帮我复审一下是否为{}，"
    "返回格式:{{'result':True or False,'according_to':'如果是你的依据是什么,如果不是你的依据又是什么,'}}"
    "返回结果以json形式"
)

def _build_prompt_by_params(model_params):
    labels = "、".join(
        p.get("paramChar", "").strip()
        for p in model_params
        if p.get("paramChar", "").strip()
    )
    return TARGET_PROMPT_TEMPLATE.format(labels, labels)

for d in algo_config:
    if not d.get("model_prompt"):
        d["model_prompt"] = _build_prompt_by_params(d.get("modelParams", []))


model_path = {d.get("algorithmNum"): {"modelPath": d.get("modelPath", ""), "withTrack": d.get("withTrack" ,False), "modelType" :d.get("modelType" ,False), "text_prompt": d.get("text_prompt", 'car'), "imgsz": d.get("imgsz", None)} for d in algo_config}

model_prompt = {d.get("algorithmNum"): d.get("model_prompt", "") for d in algo_config}

if __name__ == '__main__':
    # model_path = {d.get("algorithmNum"): {"modelPath": d.get("modelPath", ""), "withTrack": d.get("withTrack",False), "modelType":d.get("modelType",False)} for d in algo_config}
    # print(model_path)
    # a = {'car': 0.1, 'vehicle': 0.5, 'person': 0.5, 'build_rubbish': 0.5, 'live_rubbish': 0.5}
    # print(min(a.values()))
    # thresh = []
    # data = [{'algorithmNum': 1, 'paramers': [{'car': 0.5}, {'vehicle': 0.5}, {'person': 0.5}]}, {'algorithmNum': 2, 'paramers': [{'build_rubbish': 0.5}, {'live_rubbish': 0.5}]}]
    # for i in data:
    #     thresh.append(i['algorithmNum'])
    # print(thresh)
    a = model_prompt.get(82 ,"没有不显示")
    print(a)