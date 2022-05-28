import numpy as np
# import matplotlib.pyplot as plt


def get_num_per_class(gamma, N_max, num_cls, cls_index):
    if gamma==1:
        tmp_amount = N_max * (gamma**( -1 * (num_cls - 1.0 - cls_index) / (num_cls - 1.0)))
    else:
        tmp_amount = gamma * (gamma**( -1 * cls_index / (num_cls - 1.0)))
    return max(1, int(tmp_amount))


def get_dist(gamma, N_max, num_cls, labeled_frac):
    tmp_dist = [get_num_per_class(gamma, N_max, num_cls, i) for i in  range(0, num_cls)]
    if labeled_frac is None:
        return tmp_dist
    else:
        return modify_dist_by_frac(tmp_dist, labeled_frac)


def modify_dist_by_frac(dist, labeled_frac):
    if labeled_frac >=1.0:
        return dist
    dist_arr = np.array(dist)
    dist_arr = dist_arr * labeled_frac

    res_dist = np.ceil(dist_arr).astype(np.int32)
    # res_dist = np.floor(dist_arr).astype(np.int32)

    return res_dist.tolist()


def get_matched_lt_samples(gamma, N_max, num_cls, labeled_frac=None):
    samples_dist = get_dist(gamma, N_max, num_cls, labeled_frac)
    assert len(samples_dist) == num_cls
    return samples_dist


def test(test_gamma, test_n0, test_num_cls, test_frac, filename):
    cls_index = range(0, test_num_cls)
    cls_dist = get_matched_lt_samples(test_gamma, test_n0, test_num_cls, test_frac)

    print(cls_dist)

    f_size = (8,6)
    ff_name = "monospace" # Times New Roman
    myfont = {'family': ff_name, 'weight': 'normal',
              'color':  'black', 'size': 15,}

    plt.figure(figsize=f_size)
    plt.bar(cls_index, cls_dist, color = '#297083', align = 'center')
    plt.xlabel("Class Index", fontdict=myfont)
    plt.ylabel("Number", fontdict=myfont)
    plt.title(f"Class:{test_num_cls} - N0:{test_n0} - Gamma:{test_gamma} - Frac:{test_frac}")
    plt.xlim([-1, test_num_cls])
    plt.yticks(fontproperties=ff_name, size = 14)
    plt.xticks(range(0, 10), fontproperties=ff_name,size = 14)
    plt.savefig(f'{filename}-{test_gamma}-{test_n0}-{test_num_cls}-{test_frac}.jpg')
    plt.close()
    

if __name__ == "__main__":
    test_gamma = 50
    test_n0 = 5000 
    test_num_cls = 10 
    
    test_frac = None
    filename = "unlabel-test-matched"
    test(test_gamma, test_n0, test_num_cls, test_frac, filename)

    test_frac = 0.1
    filename = "label-test-matched"
    test(test_gamma, test_n0, test_num_cls, test_frac, filename)
