#!/usr/bin/python

from datetime import date, datetime, timedelta

# From http://stackoverflow.com/questions/153584/how-to-iterate-over-a-timespan-after-days-hours-weeks-and-months-in-python
def datespan(startDate, endDate, delta=timedelta(days=1)):
    currentDate = startDate
    while currentDate < endDate:
        yield currentDate
        currentDate += delta

for week in datespan(date(2011, 2, 21), date(2011, 6, 17), \
    delta=timedelta(days=7)):
    print str(week.isocalendar()[1]),   # ISO week number
    friday = week + timedelta(days=4)
    # Print formatted date from Monday to Friday.
    print "(" + str(week.strftime("%d. %B")) + " - " + str(friday.strftime("%d. %B")) +")"