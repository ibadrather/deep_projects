"""
    An application programming interface (API) is a way for two or more 
    computer programs to communicate with each other. It is a type of 
    software interface, offering a service to other pieces of software.
"""
from fastapi import FastAPI, Path, Query, HTTPException, status
from typing import Optional
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    brand: Optional[str] = None

class UpdateItem(BaseModel):
    name: Optional[str] = None
    price: Optional[float] = None
    brand: Optional[str] = None


app = FastAPI()

## We will create an endpoint now.
# An API endpoint is the point of entry in a communication channel when two 
# systems are interacting. It refers to touchpoints of the communication 
# between an API and a server.
# We have kind of a base server url: e.g; /hello, /get-item
# in this case it will be localhost because we are not deploying this app
# localhost/hello
# for facebook.com/home /home will be an endpoint 

# GET -> This endpoint will be returning information
# POST -> Sending information or creating something new
# PUT -> Modifying something that already exists
# DELETE -> Deleting

@app.get("/")
def home():
    return dict(
        Data = "Testing"
    )


@app.get("/about")
def about():
    return dict(
        Data = "About"
    )

# inventory = {
#     1 : {
#         "name": "milk",
#         "price": 3.00,
#         "brand": "ja"
#     }
# }

inventory = {}

@app.get("/get-item/{item_id}")
def get_item(item_id: int = Path(None, description="The ID of the item you would like to view ", gt=0)):
    if item_id not in inventory:
        raise HTTPException(status_code=400, detail="Item ID not found.")
    
    return inventory[item_id]


@app.get("/get-by-name")    # http://127.0.0.1:8000/get-by-name?name=milk
def get_item(name: str = Query(None, title="Name", desctiption="Name of item")):
    for item_id in inventory:
        if inventory[item_id].name == name:
            return inventory[item_id]
    
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item name not found.")


@app.post("/create-item/{item_id}")
def create_item(item_id: int,item: Item):
    if item_id in inventory:
        raise HTTPException(status_code=400, detail="Item ID already in use.")
    
    inventory[item_id] = item

    return inventory[item_id]


# to update an item

@app.put("/update-item/{item_id}")
def update_item(item_id: int, item: UpdateItem):
    if item_id not in inventory:
        raise HTTPException(status_code=400, detail="Item ID not found.")
    
    if item.name != None:
        inventory[item_id].name = item.name
    
    if item.brand != None:
        inventory[item_id].brand = item.brand
    
    if item.price != None:
        inventory[item_id].price = item.price

    return inventory[item_id]


# delete
@app.delete("/delete-item")
def delete_item(item_id: int = Query(..., description="The ID of item that needs to be deleted", gt=0)):
    if item_id not in inventory:
        raise HTTPException(status_code=400, detail="Item ID not found.")
    
    del inventory[item_id]

    return {"Sucessful": f"Item with ID = {item_id} was deleted."}


# Status Codes: 

# unicorn api:app --reload
# unicron filename_without_extension:app_name --reload everytime change is made to the file