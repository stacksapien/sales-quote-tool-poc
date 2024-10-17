from jinja2 import Environment, FileSystemLoader
import json

# JSON data
json_data = '''
{
    "client_name": "Vishal Verma",
    "client_email": "vishalv@stacksapien.com",
    "client_address": "#830, Phase-2, Ramdarbar",
    "type_of_build": "New Build",
    "budgets": [
        {
            "type": "Basic",
            "rooms": {
                "Hallway": {
                    "requirement": "Floor",
                    "products": [
                        {
                            "name": "BeoLab 50; Silver- Single Secondary",
                            "part_number": "'1620386",
                            "category": "Speakers",
                            "subcategory": "Floorstanding",
                            "type": "Floor",
                            "rating": 3,
                            "short_description": "BeoLab 50; Silver- Single Secondary",
                            "long_description": "BeoLab 50; Silver- Single Secondary",
                            "quantity": 1,
                            "unit_price": 26950,
                            "reason": ""
                        }
                    ]
                },
                "Lounge": {
                    "requirement": "Wall",
                    "products": [
                        {
                            "name": "BeoLab 50; Piano Black- Single Secondary",
                            "part_number": "'1620486",
                            "category": "Speakers",
                            "subcategory": "Floorstanding",
                            "type": "Wall",
                            "rating": 3,
                            "short_description": "BeoLab 50; Piano Black- Single Secondary",
                            "long_description": "BeoLab 50; Piano Black- Single Secondary",
                            "quantity": 1,
                            "unit_price": 26950,
                            "reason": ""
                        }
                    ]
                }
            }
        },
        {
            "type": "Premium",
            "rooms": {
                "Hallway": {
                    "requirement": "Floor",
                    "products": [
                        {
                            "name": "BeoLab 70; Silver- Single Secondary",
                            "part_number": "'1620786",
                            "category": "Speakers",
                            "subcategory": "Floorstanding",
                            "type": "Floor",
                            "rating": 4,
                            "short_description": "BeoLab 70; Silver- Single Secondary",
                            "long_description": "BeoLab 70; Silver- Single Secondary",
                            "quantity": 1,
                            "unit_price": 34950,
                            "reason": ""
                        }
                    ]
                },
                "Lounge": {
                    "requirement": "Wall",
                    "products": [
                        {
                            "name": "BeoLab 80; Piano Black- Single Secondary",
                            "part_number": "'1620886",
                            "category": "Speakers",
                            "subcategory": "Floorstanding",
                            "type": "Wall",
                            "rating": 4,
                            "short_description": "BeoLab 80; Piano Black- Single Secondary",
                            "long_description": "BeoLab 80; Piano Black- Single Secondary",
                            "quantity": 1,
                            "unit_price": 34950,
                            "reason": ""
                        }
                    ]
                }
            }
        },
        {
            "type": "Ultimate",
            "rooms": {
                "Hallway": {
                    "requirement": "Floor",
                    "products": [
                        {
                            "name": "BeoLab 100; Silver- Single Secondary",
                            "part_number": "'16210086",
                            "category": "Speakers",
                            "subcategory": "Floorstanding",
                            "type": "Floor",
                            "rating": 5,
                            "short_description": "BeoLab 100; Silver- Single Secondary",
                            "long_description": "BeoLab 100; Silver- Single Secondary",
                            "quantity": 1,
                            "unit_price": 49950,
                            "reason": ""
                        }
                    ]
                },
                "Lounge": {
                    "requirement": "Wall",
                    "products": [
                        {
                            "name": "BeoLab 120; Piano Black- Single Secondary",
                            "part_number": "'16212086",
                            "category": "Speakers",
                            "subcategory": "Floorstanding",
                            "type": "Wall",
                            "rating": 5,
                            "short_description": "BeoLab 120; Piano Black- Single Secondary",
                            "long_description": "BeoLab 120; Piano Black- Single Secondary",
                            "quantity": 1,
                            "unit_price": 49950,
                            "reason": ""
                        }
                    ]
                }
            }
        }
    ]
}
'''

# Load JSON data
data = json.loads(json_data)

# Setup Jinja2 environment
file_loader = FileSystemLoader('.')
env = Environment(loader=file_loader)

# Load the template
template = env.get_template('template.html')

# Render the template with dynamic data
output = template.render(client_name=data['client_name'],
                         client_email=data['client_email'],
                         client_address=data['client_address'],
                         type_of_build=data['type_of_build'],
                         budgets=data['budgets'])

# Save the rendered HTML to a file
with open('output.html', 'w') as f:
    f.write(output)

print("HTML generated successfully!")
