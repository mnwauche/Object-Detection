labels_path = 'mscoco_label_map.pbtxt'
file = open(labels_path, 'r')
labels_str = file.read()
labels = {}
s = 0
while labels_str.find('{', s) != -1:
    start = labels_str.find('{', s)
    id_start = labels_str.find('id', s) + 4
    id_end = labels_str.find('\n', id_start)
    dname = labels_str.find('display_name', start)
    dname_start = labels_str.find('\"', dname)
    dname_end = labels_str.find('\"', dname_start + 1)
    end = labels_str.find('}', start + 1)
    labels[int(labels_str[id_start:id_end])] = labels_str[dname_start + 1:dname_end]
    print(labels_str[id_start:id_end] + ' ' + labels_str[dname_start + 1:dname_end])
    s = end
    
file.close()    
file = open('parsed_labels.txt', 'w')
file.write(str(labels))
file.close()