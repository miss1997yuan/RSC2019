
test=True
if test:
        # 定义参数
        NUMS = 5000
        # action_type的种类 最大长度
        n_action = 10
        impression_length = 15
        # 交互序列的种类 以及最大的长度
        n_items=1000
        #用户浏览序列最大长度
        item_series_length = 25
        # 酒店的信息的纬度 一个1000个酒店让后每个酒店55个纬度，10中交互类型
        item_info_dim = 55

        batch_size=100

        embedding_size=120

        num_hidden=130

        epoches=30

        dropout_p_hidden=0.3

        STEP=10

        logs_path='./checkpoint'

        DATA_SIZE=5000

        epoch_step=DATA_SIZE//batch_size

        MODEL_NAME='gru-model'

else:
        pass
