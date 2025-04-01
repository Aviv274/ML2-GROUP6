import os
from typing import Optional

import serpapi
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
class FlightsInput(BaseModel):
    departure_airport: Optional[str] = Field(description='Departure airport code (IATA)')
    arrival_airport: Optional[str] = Field(description='Arrival airport code (IATA)')
    outbound_date: Optional[str] = Field(description='Parameter defines the outbound date. The format is YYYY-MM-DD. e.g. 2024-06-22')
    return_date: Optional[str] = Field(description='Parameter defines the return date. The format is YYYY-MM-DD. e.g. 2024-06-28')
    adults: Optional[int] = Field(1, description='Parameter defines the number of adults. Default to 1.')
    children: Optional[int] = Field(0, description='Parameter defines the number of children. Default to 0.')



class FlightsInputSchema(BaseModel):
    params: FlightsInput


@tool(args_schema=FlightsInputSchema)
def flights_finder(params: FlightsInput):
    '''
    Find flights using the Google Flights engine.

    Returns:
        dict: Flight search results.
    '''

    params = {
        'api_key': os.getenv('SERPAPI_API_KEY'),
        'engine': 'google_flights',
        'hl': 'en',
        'gl': 'us',
        'departure_id': params.departure_airport,
        'arrival_id': params.arrival_airport,
        'outbound_date': params.outbound_date,
        'return_date': params.return_date,
        'currency': 'USD',
        'adults': params.adults,
        'stops': '1',
        'children': params.children
    }

    search = serpapi.search(params)
    results = search.data
    return results['best_flights'][:5]



