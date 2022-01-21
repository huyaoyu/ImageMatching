
import argparse
import cv2
import glob
import json
import sys
import os
import re
from termcolor import colored

from CommonPython.Filesystem import Filesystem

from proxies.register import ( PROXIES, make_object )
from homography.homography_ocv import HomographyCalculator
from homography.homography_scale import scale_homography_matrix
from visualization.draw_matches import ( 
    draw_matches_ocv, draw_matches_plt )
from visualization.merge import merge_two_single_channels

# Global regex pattern.
RE_C = re.compile(r'-(\d+).jpg$')

def print_error(msg):
    print(colored(msg, 'red'))

def find_template_filenames(t_dir):
    global RE_C

    # Find all the .jpg files.
    files = sorted( glob.glob( os.path.join(t_dir, '*.jpg') ) )
    assert( len(files) > 0 ), f'No files found in {t_dir}'

    # Check the filenames.
    templates = dict()
    for f in files:
        matches = RE_C.search(f)
        if ( matches is None ):
            # Not found.
            continue
        
        templates[ matches[1] ] = f

    assert ( len(templates) > 0 ), \
        f'No valid templates found in {t_dir}'

    return templates

def read_image(fn):
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '

    return cv2.imread(fn, cv2.IMREAD_UNCHANGED)

def read_templates(template_fn_dict):
    templates = dict()
    for k, v in template_fn_dict.items():
        print(f'Read {v}')
        t_img = read_image(v)
        templates[k] = dict( fn=v, img=t_img )

    return templates

def find_input_filenames(in_dir):
    # Find all the .jpg files.
    files = sorted( glob.glob( os.path.join(in_dir, '*.jpg'), recursive=True ) )
    assert( len(files) > 0 ), f'No files found in {in_dir}'

    return files

def create_proxy(proxy_conf):
    return make_object(PROXIES, proxy_conf)

def write_result_json(fn, res_dict):
    with open(fn, 'w') as fp:
        json.dump(res_dict, fp, indent=1)

def process(proxy, hc, fn, templates, out_dir, write_matched_image=False):
    global RE_C

    print(fn)

    # Check the filename.
    matches = RE_C.search(fn)
    if (matches is None):
        print_error(f'{fn} is an invalid file name. No template is specified as filename suffix. ')
        return False

    # Valid filename.
    template_idx = matches[1]

    # Read the input file.
    try:
        img = read_image(fn)
    except Exception as exc:
        print_error(f'Read {fn} failed. ')
        print_error(f'Exception: {str(exc)}')
        return False

    # Prepare the input for the proxy.
    try:
        inputs = dict(
            img_0=templates[template_idx]['img'],
            img_1=img,
        )
    except KeyError:
        print_error(f'Index {template_idx} is not in the templates. ')
        return False

    # Feture extraction.
    outputs = proxy(inputs)

    # Compute the homography.
    coord_0 = outputs['coord_0']
    coord_1 = outputs['coord_1']

    if ( coord_0.shape[0] < 4 ):
        print_error(f'Template {templates[template_idx]["fn"]} has less than 4 feature points. ')
        return False

    if ( coord_1.shape[0] < 4 ):
        print_error(f'Input {fn} has less than 4 feature points. ')
        return False

    if ( proxy.task_type == 'extraction' ):
        desc_0  = outputs['desc_0']
        desc_1  = outputs['desc_1']

        h_mat, good_matches = \
            hc( coord_1.reshape((-1, 1, 2)), coord_0.reshape((-1, 1, 2)), desc_1, desc_0 )
        if ( h_mat is not None ):
            n_good_matches=len(good_matches)
    elif ( proxy.task_type == 'matching' ):
        # import ipdb; ipdb.set_trace()
        # Prepare the keypoints.
        h_mat, h_mask = hc.compute_homography_by_matched_results( 
            coord_1.reshape((-1, 1, 2)), coord_0.reshape((-1, 1, 2)) )

        n_good_matches=len(coord_0)
    else:
        print_error(f'Unexpected proxy.task_type: {proxy.task_type}. ')
        return False

    if ( h_mat is None ):
        print_error(f'Failed to compute homographyu for input {fn}. ')
        return False

    # Rescale h_mat.
    if ( proxy.ori_shape != proxy.new_shape ):
        h_mat = scale_homography_matrix( 
            h_mat, 
            proxy.new_shape, proxy.ori_shape, 
            proxy.new_shape, proxy.ori_shape )

    print(h_mat)

    # Save the result.
    res_dict = dict(
        fn=fn,
        template_fn=templates[template_idx]['fn'],
        h_mat=h_mat.reshape((-1,)).tolist(),
        n_good_matches=n_good_matches,
    )
    parts = Filesystem.get_filename_parts(fn)
    out_fn = os.path.join( out_dir, f'{parts[1]}_dingwei.json' )
    try:
        write_result_json(out_fn, res_dict)
    except Exception as exc:
        print_error(f'Write results of {fn} failed. ')
        print_error(f'Exception: {str(exc)}')
        return False

    if ( write_matched_image ):
        img_0_scaled = cv2.resize( inputs['img_0'], ( proxy.new_shape[1], proxy.new_shape[0] ), interpolation=cv2.INTER_CUBIC )
        img_1_scaled = cv2.resize( inputs['img_1'], ( proxy.new_shape[1], proxy.new_shape[0] ), interpolation=cv2.INTER_CUBIC )

        if ( proxy.task_type == 'extraction' ):
            img_matches = draw_matches_ocv( 
                img_1_scaled, img_0_scaled, 
                coord_1, coord_0, good_matches )
        else:
            img_matches = draw_matches_plt( 
                img_1_scaled, img_1_scaled, 
                coord_1, coord_0, outputs['confidence'])

        out_fn = os.path.join(out_dir, f'{parts[1]}_matches.jpg')
        cv2.imwrite(out_fn, img_matches)

        # Merge.
        # Warp the test/source image.
        warped = cv2.warpPerspective(
            img_1_scaled, h_mat, ( img_1_scaled.shape[1], img_1_scaled.shape[0] ), flags=cv2.INTER_LINEAR)

        # Merge the reference/destination and test/source images.
        merged = merge_two_single_channels( img_0_scaled, warped )

        out_fn = os.path.join(out_dir, f'{parts[1]}_merged.jpg')
        cv2.imwrite(out_fn, merged)

    return True

def handle_args():
    parser = argparse.ArgumentParser(description='Run image matching. ')

    parser.add_argument('workdir', type=str, 
        help='The working directory. ')

    parser.add_argument('templatedir', type=str, 
        help='The template directory. ')

    parser.add_argument('indir', type=str, 
        help='The directory of the input images that are to be processed.')

    parser.add_argument('outdir', type=str, 
        help='The output directory. ')
    
    return parser.parse_args()

def main():
    # Get the input arguments.
    args = handle_args()

    # Prepare the output directory.
    Filesystem.test_directory(args.outdir)

    # Find all the input files.
    input_fns = find_input_filenames(args.indir)
    print(f'{len(input_fns)} input files found. ')

    # Find and read the template files.
    template_fns = find_template_filenames(args.templatedir)
    templates = read_templates(template_fns)

    # Read the proxy configuration file.
    sys.path.insert(0, args.workdir)
    from input_conf import conf

    # Create and initialize the proxy.
    proxy = create_proxy(conf['proxy'])
    proxy.initialize()

    # Create a homography calculator.
    hc = HomographyCalculator()

    # Process the input files.
    for f in input_fns:
        process( proxy, hc, f, templates, args.outdir, write_matched_image=True )

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
