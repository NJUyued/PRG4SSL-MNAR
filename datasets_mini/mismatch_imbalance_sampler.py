import numpy as np
# import matplotlib.pyplot as plt

def get_num_per_class(gamma, N0, num_cls, cls_index):
    tmp_amount =  N0*(gamma** (-cls_index /num_cls))
#     tmp_amount =  N0*(gamma** (-cls_index))
    return max(1, int(tmp_amount))

def modify_dist_by_amount(dist, bounded_num):
#     print(dist)
#     return dist
    remain = sum(dist) - bounded_num
#     print(remain)
    if remain == 0:
        return dist
    if remain > 0:
        while True:
            for i in range(1, len(dist)-2):
                dist[i] = max(dist[i] - 1, 1)
                remain = remain - 1
                if remain == 0:
                    break
            remain = sum(dist) - bounded_num
            if remain == 0:
                break
        return dist
    if remain < 0:
        for i in range(1, len(dist)-2):
            dist[i] = dist[i] + 1
            remain = remain + 1
            if remain == 0:
                break
        return dist

def get_dist(gamma, N0, num_cls, bounded_num):
    tmp_dist = [get_num_per_class(gamma, N0, num_cls,i) for i in  range(0, num_cls)]
#     return tmp_dist
    return modify_dist_by_amount(tmp_dist, bounded_num)

def find_proper_gamma(N0, num_cls, bounded_num, from_index=1, to_index=100000):
    min_var = -1
    min_sum = -1
    max_var = -1
    max_sum = -1
    for gamma in range(from_index, to_index):
        tmp_dist = [get_num_per_class(gamma, N0, num_cls,i) for i in  range(0, num_cls)]
        tmp_sum = sum(tmp_dist)
        if tmp_sum > bounded_num:
            if max_var < gamma:
                max_var = gamma
                max_sum = tmp_sum
        else:
            min_var = gamma
            min_sum = tmp_sum
            break
    if abs(min_sum-bounded_num) < abs(max_sum-bounded_num):
        desire_var = min_var
    else:
        desire_var = max_var
    return min_var, min_sum, max_var,max_sum, desire_var

def plot_dist_wt_no(gamma_lst, N_lst, num_cls, bounded_num):

    num_plots = len(N_lst)
    cols = (num_plots)**0.5 + 2
    cols = int(cols)
    rows = num_plots // cols if num_plots % cols ==0 else num_plots // cols + 1
    
    
    tt = int(10*rows/cols)
    plt.figure(figsize=(10,tt))
    class_index = list(range(num_cls))
    for i, each_n in enumerate(N_lst):
        gamma = gamma_lst[i]
        tmp_out = get_dist(gamma, each_n, num_cls, bounded_num)
        total_amount = sum(tmp_out)
        
        imb_ratio = max(tmp_out) / min(tmp_out)
        
        ax = plt.subplot(rows, cols, i+1)
        ax.bar(class_index, tmp_out, color = '#297083', align = 'center')
        ax.set_title("No={} - {:.2f}".format(each_n, imb_ratio))
        ax.set_ylabel(f"Num={total_amount}")
        
    plt.tight_layout()
    plt.show()


def get_mismatched_imbalance_samples(N0, num_cls, bounded_num):
    find_res = find_proper_gamma(N0, num_cls, bounded_num)
    gamma = find_res[-1]
    samples_dist = get_dist(gamma, N0, num_cls, bounded_num)
    assert len(samples_dist) == num_cls
    return samples_dist


def test(test_n0, test_num_cls, test_bound, filename):
    cls_index = range(0, test_num_cls)
    cls_dist = get_mismatched_imbalance_samples(test_n0, test_num_cls, test_bound)

    print(cls_dist)

    f_size = (8,6)
    ff_name = "monospace" # Times New Roman
    myfont = {'family': ff_name, 'weight': 'normal',
              'color':  'black', 'size': 15,}

    plt.figure(figsize=f_size)
    plt.bar(cls_index, cls_dist, color = '#297083', align = 'center')
    plt.xlabel("Class Index", fontdict=myfont)
    plt.ylabel("Number", fontdict=myfont)
    plt.title(f"Class:{test_num_cls} - Total:{test_bound}/{sum(cls_dist)} - N0:{test_n0}")
    plt.xlim([-1, test_num_cls])
    plt.yticks(fontproperties=ff_name, size = 14)
    plt.xticks(range(0, 10), fontproperties=ff_name,size = 14)
    plt.savefig(f'{filename}-{test_n0}-{test_num_cls}-{test_bound}.jpg')
    plt.close()

if __name__ == "__main__":
    test_n0 = 10
    test_num_cls = 10
    test_bound = 40
    filename="test"
    test(test_n0, test_num_cls, test_bound, filename)
    