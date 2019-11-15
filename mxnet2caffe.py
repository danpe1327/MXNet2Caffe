import os

os.environ['GLOG_minloglevel'] = '2'

import json
import sys
import argparse
import find_mxnet
import find_caffe
import mxnet as mx
import caffe


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
    parser.add_argument('--mx-model', type=str, default='model_mxnet/facega,0')
    parser.add_argument('--cf-prototxt', type=str, default='model_caffe/facega.prototxt')

    args = parser.parse_args()

    return args


def mxnet2caffe(args):
    # ------------------------------------------
    # Load
    vec = args.mx_model.split(',')
    assert len(vec) > 1
    prefix = vec[0]
    epoch = int(vec[1])
    print('loading', prefix, epoch)

    # load mxnet model
    _, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    mx_json = '%s-symbol.json' % prefix
    with open(mx_json) as json_file:
        jdata = json.load(json_file)

    # load caffe net
    net = caffe.Net(args.cf_prototxt, caffe.TRAIN)

    # ------------------------------------------
    # Convert
    all_keys = arg_params.keys() + aux_params.keys()
    all_keys.sort()

    print('----------------------------------\n')
    print('ALL KEYS IN MXNET:')
    # print(all_keys)
    print('%d KEYS' % len(all_keys))

    print('----------------------------------\n')
    print('VALID KEYS:')
    for i_key, key_i in enumerate(all_keys):
        try:
            if 'data' == key_i:
                pass
            elif '_weight' in key_i:
                key_caffe = key_i.replace('_weight', '')

                if key_caffe not in net.params:  # add for mnet-retinaface
                    key_caffe = key_caffe + '_fwd'

                if 'fc' in key_i:
                    print(key_i)
                    print(arg_params[key_i].shape)
                    print(net.params[key_caffe][0].data.shape)
                net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
            elif '_bias' in key_i:
                key_caffe = key_i.replace('_bias', '')
                net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
            elif '_gamma' in key_i and 'relu' not in key_i:
                # for mxnet batchnorm layer, if fix_gamma == 'True', the values should be 1.
                fix_gamma_param = False
                for layer in jdata['nodes']:
                    if layer['name'] == key_i:
                        if 'attrs' in layer and 'fix_gamma' in layer['attrs'] and str(layer['attrs']['fix_gamma']) == 'True':
                            fix_gamma_param = True
                        else:
                            fix_gamma_param = False
                        break

                key_caffe = key_i.replace('_gamma', '_scale')

                if key_caffe not in net.params:  # add for mnet-retinaface
                    key_caffe = key_caffe.replace('_scale', '_fwd_scale')

                if fix_gamma_param:
                    net.params[key_caffe][0].data[...] = 1
                else:
                    net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat

            # TODO: support prelu
            elif '_gamma' in key_i and 'relu' in key_i:  # for prelu
                key_caffe = key_i.replace('_gamma', '')
                assert (len(net.params[key_caffe]) == 1)
                net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
            elif '_beta' in key_i:
                key_caffe = key_i.replace('_beta', '_scale')

                if key_caffe not in net.params:  # add for mnet-retinaface
                    key_caffe = key_caffe.replace('_scale', '_fwd_scale')

                net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
            elif '_moving_mean' in key_i:
                key_caffe = key_i.replace('_moving_mean', '')
                net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
                net.params[key_caffe][2].data[...] = 1
            elif '_moving_var' in key_i:
                key_caffe = key_i.replace('_moving_var', '')
                net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
                net.params[key_caffe][2].data[...] = 1

            elif '_running_mean' in key_i:
                key_caffe = key_i.replace('_running_mean', '_fwd')
                net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
                net.params[key_caffe][2].data[...] = 1
            elif '_running_var' in key_i:
                key_caffe = key_i.replace('_running_var', '_fwd')
                net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
                net.params[key_caffe][2].data[...] = 1

            else:
                sys.exit("Warning!  Unknown mxnet:{}".format(key_i))

            print("% 3d | %s -> %s, initialized." % (i_key, key_i.ljust(40), key_caffe.ljust(30)))

        except KeyError:
            print("\nError!  key error mxnet:{}".format(key_i))

    # Finish
    cf_model = args.cf_prototxt.replace('prototxt', 'caffemodel')
    net.save(cf_model)
    print("\n- Finished.\n")


if __name__ == '__main__':
    args = parse_args()
    mxnet2caffe(args)
