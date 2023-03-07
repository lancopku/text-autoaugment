# Policy found on wiki_qa
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .augmentation import augment_list


def remove_deplicates(policies):
    s = set()
    new_policies = []
    for ops in policies:
        key = []
        for op in ops:
            key.append(op[0])
        key = '_'.join(key)
        if key in s:
            continue
        else:
            s.add(key)
            new_policies.append(ops)

    return new_policies


def policy_decoder(augment, num_policy, num_op):
    '''

    :param augment: related to dict `space` and `config` in `exp_config`
    :param num_policy: num of policies, default 5?
    :param num_op: num of operations per policy, default 2?
    :return: #num_policy sub-policies, each sub-policies contains #num_op operations
    '''
    op_list = augment_list()
    policies = []
    for i in range(num_policy):
        ops = []
        for j in range(num_op):
            op_idx = augment['policy_%d_%d' % (i, j)]
            op_prob = augment['prob_%d_%d' % (i, j)]
            op_level = augment['level_%d_%d' % (i, j)]
            ops.append((op_list[op_idx][0].__name__, op_prob, op_level))
        policies.append(ops)
    return policies


def yelp2():
    return [[('tfidf_word_substitute', 0.9512914186004635, 0.5111819132642521),
             ('random_word_delete', 0.08180101348367774, 0.979763746611628)],
            [('random_word_swap', 0.5655615885534724, 0.8180667343235334),
             ('random_word_delete', 0.8537622143927265, 0.03893147547204692)],
            [('random_word_swap', 0.30385510347920847, 0.3049506130528053),
             ('tfidf_word_substitute', 0.5486285112912943, 0.10281387222638078)],
            [('tfidf_word_substitute', 0.6295169826415187, 0.6803876702537599),
             ('synonym_word_substitute', 0.12909240997304433, 0.5693145762533841)],
            [('tfidf_word_insert', 0.07144648407889143, 0.8676263900036898),
             ('random_word_swap', 0.19076051045929354, 0.9214686548873446)],
            [('synonym_word_substitute', 0.3530966586851968, 0.4254413492543815),
             ('tfidf_word_substitute', 0.5511222962678954, 0.27667619702350865)],
            [('random_word_delete', 0.6132907651014691, 0.5436974611466988),
             ('tfidf_word_substitute', 0.4369111959906028, 0.014370013695976791)],
            [('tfidf_word_substitute', 0.6577030920789507, 0.6948986804714572),
             ('tfidf_word_insert', 0.5971443584631903, 0.330365511768527)],
            [('synonym_word_substitute', 0.46529762149432813, 0.8661779845852324),
             ('random_word_delete', 0.21776020319762318, 0.6619847428092126)],
            [('synonym_word_substitute', 0.3562832725772521, 0.9948754750085635),
             ('synonym_word_substitute', 0.3661529937591429, 0.10598350701208797)],
            [('random_word_swap', 0.6190693462639046, 0.1158871610234205),
             ('synonym_word_substitute', 0.544700810678522, 0.2859493613750108)],
            [('tfidf_word_substitute', 0.8608454060064963, 0.5323391079618531),
             ('tfidf_word_insert', 0.40068472332275223, 0.16975377422326549)]]


def yelp5():
    return [[('random_word_delete', 0.33786297476526733, 0.6434710505517958),
             ('synonym_word_substitute', 0.6936147188511684, 0.3894437187016606)],
            [('random_word_delete', 0.700781165698797, 0.3075995672409439),
             ('random_word_delete', 0.8460075076488855, 0.5410548678348011)],
            [('tfidf_word_substitute', 0.7216035297769526, 0.7008316910874055),
             ('random_word_delete', 0.5826407943783878, 0.9332048588438929)],
            [('random_word_swap', 0.4868789562278112, 0.441818085706321),
             ('tfidf_word_substitute', 0.720081206002785, 0.9964875289266723)],
            [('random_word_delete', 0.5274137192428784, 0.5346501769570986),
             ('random_word_delete', 0.9268931122884394, 0.6561391337445356)],
            [('random_word_delete', 0.716008076640467, 0.33692499878278426),
             ('random_word_swap', 0.7788337655050918, 0.5516195739629717)],
            [('tfidf_word_substitute', 0.5794431046827291, 0.624757169610185),
             ('tfidf_word_substitute', 0.5319319425673782, 0.39489559886961234)],
            [('tfidf_word_insert', 0.5806063715877541, 0.468247737805887),
             ('tfidf_word_insert', 0.3980064660940441, 0.29310282182122194)],
            [('tfidf_word_insert', 0.6014340751762324, 0.47502110071065984),
             ('random_word_delete', 0.3866124077451312, 0.7583822538496063)],
            [('random_word_swap', 0.676800061447225, 0.7133596684872278),
             ('tfidf_word_substitute', 0.5453524855682802, 0.6816959632891163)],
            [('random_word_swap', 0.5458497905022153, 0.9226046939000903),
             ('random_word_delete', 0.8857207978573708, 0.6967199442402441)]]


def imdb():
    return [[('tfidf_word_substitute', 0.7701989738031709, 0.1413189624238933),
             ('tfidf_word_substitute', 0.5017627182356647, 0.6120126108801994)],
            [('random_word_delete', 0.723773044425342, 0.09343851538354664),
             ('tfidf_word_insert', 0.65871023661296, 0.18515171972877364)],
            [('tfidf_word_insert', 0.6855258510775246, 0.25155717455902515),
             ('random_word_swap', 0.5855398958576279, 0.8038331622764382)],
            [('tfidf_word_substitute', 0.766621630113518, 0.9996972209388315),
             ('tfidf_word_substitute', 0.35757492431715654, 0.5094646924565402)],
            [('tfidf_word_insert', 0.7201447184878109, 0.27394476379211474),
             ('tfidf_word_insert', 0.7102010737993143, 0.7735198668033543)],
            [('random_word_swap', 0.2602481801512967, 0.1094817565926513),
             ('random_word_swap', 0.377657390084737, 0.5860711221450985)],
            [('random_word_swap', 0.3628204769855412, 0.4174830136614628),
             ('tfidf_word_substitute', 0.7449752005673833, 0.5806015818156878)],
            [('random_word_delete', 0.6869314076454281, 0.30212959898754665),
             ('random_word_delete', 0.7791546348246594, 0.6329609313430418)],
            [('random_word_delete', 0.07837947214430199, 0.37349725564746655),
             ('synonym_word_substitute', 0.6222849277805101, 0.6520156160496825)],
            [('tfidf_word_insert', 0.6448512230749102, 0.007104243636811641),
             ('random_word_swap', 0.45273525613734494, 0.4978187905095047)],
            [('random_word_delete', 0.44394804096384133, 0.037671545592863925),
             ('tfidf_word_substitute', 0.6899616027426003, 0.8021392220253207)]]


def sst5():
    return [[('random_word_delete', 0.4376604353412977, 0.44043253964304485),
             ('synonym_word_substitute', 0.5875942027594719, 0.998514395402531)],
            [('tfidf_word_insert', 0.6565166227516329, 0.21330789076297246),
             ('tfidf_word_substitute', 0.11299312933023159, 0.49719311238906505)],
            [('synonym_word_substitute', 0.6861150131708554, 0.17491373112783754),
             ('synonym_word_substitute', 0.4079121596956716, 0.14994998127844966)],
            [('tfidf_word_substitute', 0.6684155726744387, 0.03639046668649182),
             ('synonym_word_substitute', 0.987814822565414, 0.4421747192163583)],
            [('synonym_word_substitute', 0.5452347978175406, 0.5828686337771589),
             ('random_word_delete', 0.5427887439733309, 0.7797002829854326)],
            [('random_word_swap', 0.07941610178776508, 0.35601684106539533),
             ('random_word_swap', 0.35327541200910634, 0.27591459199458696)],
            [('tfidf_word_insert', 0.5021626674035361, 0.24326437485991964),
             ('synonym_word_substitute', 0.3363344567219857, 0.22609807681108454)],
            [('synonym_word_substitute', 0.7010453890387791, 0.09848782175096363),
             ('synonym_word_substitute', 0.8458897618930373, 0.35062068730929496)],
            [('synonym_word_substitute', 0.35191398074072006, 0.4878134393552393),
             ('random_word_delete', 0.6170268081935109, 0.8868930630258844)],
            [('tfidf_word_substitute', 0.059174010617455586, 0.40698841480571085),
             ('random_word_swap', 0.4679461899402404, 0.7487390093572956)],
            [('tfidf_word_insert', 0.7282101888360597, 0.033082661180942996),
             ('synonym_word_substitute', 0.27061956724444475, 0.14898751991461168)],
            [('tfidf_word_substitute', 0.5025015728914708, 0.15459943245089142),
             ('tfidf_word_substitute', 0.7412162578055561, 0.20250875996527518)]]


def trec():
    return [[('synonym_word_substitute', 0.7492730962660217, 0.8816452863413866),
             ('synonym_word_substitute', 0.33184334794125936, 0.5208169910984721)],
            [('random_word_swap', 0.603927626550949, 0.03168123331963181),
             ('synonym_word_substitute', 0.5867758555030048, 0.7287574769015046)],
            [('tfidf_word_insert', 0.5945610605693987, 0.34278286156354293),
             ('synonym_word_substitute', 0.6347655416328688, 0.6778017258574649)],
            [('tfidf_word_substitute', 0.22489968497333096, 0.7330245174712894),
             ('random_word_swap', 0.9597228128295471, 0.059212246046256634)],
            [('random_word_delete', 0.4203452055085861, 0.22356779679195654),
             ('tfidf_word_substitute', 0.8767617467184782, 0.3722966169711849)],
            [('random_word_swap', 0.45804224314084996, 0.24582533503242737),
             ('tfidf_word_insert', 0.7100281507903158, 0.9189445839700638)],
            [('random_word_swap', 0.8839667299762286, 0.45276342993947843),
             ('synonym_word_substitute', 0.8328421526598614, 0.7228649899668298)],
            [('synonym_word_substitute', 0.024628867286349748, 0.16233357796074102),
             ('random_word_swap', 0.9756625000571791, 0.46408669166135974)],
            [('random_word_delete', 0.6536868778944346, 0.14924007264240377),
             ('tfidf_word_substitute', 0.6414854813123402, 0.2006140680274639)],
            [('random_word_swap', 0.47596775863886953, 0.12078399579662603),
             ('tfidf_word_insert', 0.5605668428551196, 0.8275280745990681)],
            [('synonym_word_substitute', 0.06761427481801857, 0.32908845524510716),
             ('tfidf_word_insert', 0.9572797299031379, 0.5632362843917429)]]


policy_map = {'imdb': imdb(), 'sst5': sst5(), 'trec': trec(), 'yelp2': yelp2(), 'yelp5': yelp5(), 'custom_data': imdb(), 'sst2': sst5()}
