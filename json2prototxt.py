import os
import argparse
import json
from prototxt_basic import write_node

parser = argparse.ArgumentParser(description='Convert MXNet jason to Caffe prototxt')
parser.add_argument('--mx-json', type=str, default='./model_mxnet/facega-symbol.json')
parser.add_argument('--cf-prototxt', type=str, default='./model_caffe/facega.prototxt')
args = parser.parse_args()

with open(args.mx_json) as json_file:
    jdata = json.load(json_file)

with open(args.cf_prototxt, "w") as prototxt_file:
    bottom_of_flatten = None
    for i_node in range(0, len(jdata['nodes'])):
        node_i = jdata['nodes'][i_node]
        if str(node_i['op']) == 'null' and str(node_i['name']) != 'data':
            continue

        if str(node_i['op']) == '_copy' or str(node_i['op']) == '_minus_scalar' or str(node_i['op']) == '_mul_scalar':
            continue

        print('{}, \top:{}, name:{} -> {}'.format(i_node, node_i['op'].ljust(20),
                                                  node_i['name'].ljust(30),
                                                  node_i['name']).ljust(20))
        info = node_i

        info['top'] = info['name']
        info['bottom'] = []
        info['params'] = []
        for input_idx_i in node_i['inputs']:
            input_i = jdata['nodes'][input_idx_i[0]]

            if str(input_i['op']) == '_mul_scalar':
                info['bottom'].append('data')
            elif str(input_i['op']) == 'Dropout':
                # current_input = input_i['inputs']
                # current_input = current_input[0]
                # input_idx = jdata['nodes'][current_input[0]]
                info['bottom'].append(str(input_i['name']))
            elif str(input_i['op']) == 'Flatten':
                bottom_of_flatten = input_i['bottom']
            elif str(input_i['op']) != 'null' or (str(input_i['name']) == 'data'):
                info['bottom'].append(str(input_i['name']))

            if str(input_i['op']) == 'null':
                info['params'].append(str(input_i['name']))
                if not str(input_i['name']).startswith(str(node_i['name'])):
                    print('           use shared weight -> %s' % str(input_i['name']))
                    info['share'] = True

        if bottom_of_flatten is not None:
            info['bottom'] = bottom_of_flatten
            bottom_of_flatten = None
        write_node(prototxt_file, info)
