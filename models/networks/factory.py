from bootstrap.lib.options import Options
from models.networks.itmbf_net import ITMBFNet
from models.networks.itbf_net import ITBFNet
from models.networks.SAN import SANModel
from models.networks.BAN import BanModel
from models.networks.TDA import TDAModel
from modules.language_model import WordEmbedding, QuestionEmbedding
from modules.classifier import SimpleClassifier
from modules.visual_model import resnet50


def factory(engine):
    mode = list(engine.dataset.keys())[0]  # 'train'/'test'
    dataset = engine.dataset[mode]
    opt = Options()['model.network']
    op = 'n'  # if 'c' in op, wordembedding needed to be copied for bidirection text embedding.
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, op)
    num_hid = opt['txt_enc']['dim']  # the dimension of the output, ie. the dimension of textual feature vector
    t_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
    # 加载图像特征提取模型resnet50
    cnn_model = resnet50()
    num_classes = dataset.num_ans_candidates
    if opt['name'] == 'itmbf_net':
        input_dim = opt['attention']['v_fusion']['output_dim']
        classif = SimpleClassifier(input_dim, 256, num_classes, .5)
        net = ITMBFNet(
            w_emb=w_emb,
            t_emb=t_emb,
            img_enc=cnn_model,
            classif=classif,
            attention=opt['attention']
            )
    elif opt['name'] == 'itbf_net':
        input_dim = opt['attention']['v_fusion']['output_dim']
        classif = SimpleClassifier(input_dim, 256, num_classes, .5)
        net = ITBFNet(
            w_emb=w_emb,
            t_emb=t_emb,
            img_enc=cnn_model,
            classif=classif,
            fusion=opt['attention']
            )
    elif opt['name'] == 'san':
        input_dim = opt['attention']['output_dim']
        classif = SimpleClassifier(input_dim, 256, num_classes, .5)
        net = SANModel(
            w_emb=w_emb,
            t_emb=t_emb,
            img_enc=cnn_model,
            classif=classif,
            attention=opt['attention']
            )
    elif opt['name'] == 'ban':
        input_dim = opt['attention']['output_dim']
        classif = SimpleClassifier(input_dim, 256, num_classes, .5)
        net = BanModel(
            w_emb=w_emb,
            t_emb=t_emb,
            img_enc=cnn_model,
            classif=classif,
            attention=opt['attention']
        )
    else:
        raise ValueError()
    return net
