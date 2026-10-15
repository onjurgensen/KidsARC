import json

data = json.load(open('data/matrices_sql.json', 'rb'))[2]['data']

new_data = []
for item in data:
    if item['concept'] in ['other', 'top_and_bottom_2d']:
        continue

    if item['concept'] == 'top_and_bottom_3d':
        item['concept'] = 'foreground_and_background'
    if item['concept'] == 'above_and_below':
        item['concept'] = 'keep_above_or_below'
    if item['concept'] == 'filled_and_not_filled':
        item['concept'] = 'fill_in'

    #TODO
    # if item['id'] in ['JVyGSu', 'W08yyc', 'w1KvbB']:
    #     item['concept'] = 'move_inside'
    # if item['id'] in ['GjqiSH', 'GPAssO	', 'F42Xxc']:
    #     item['concept'] = 'keep_inside'
        
    item.pop('create_date')
    item.pop('D_Matrix')

    item['xdim'] = int(len(item['A']) ** 0.5)
    item['ydim'] = int(len(item['A']) ** 0.5)
    
    new_data.append(item)

json.dump(new_data, open('data/ambigous_arc.json', 'w'), indent=4)

