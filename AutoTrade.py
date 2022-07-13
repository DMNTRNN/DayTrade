Python Viewer, Formatter, Editor
Related Tools
Perl Beautify
Php Beautify
Ruby Beautify
Sql Beautify
Xml Beautify
398
​
399
coinlist = ['XRP', 'ETH']
400
​
401
preds = []
402
​
403
model = keras.models.load_model('model.h5', custom_objects = {
404
    'PASS': PASS
405
})
406
​
407
USDTBAL = 1000
408
trigger = 0
409
counter = 0
410
​
411
​
412
​
413
# reproducing the same trade conditions that were used
414
for simulation.Threshold value
415
for closing the order is 0.575 %
416
    while True:
417
​
418
    if trigger == 0:
419
    print(USDTBAL)
420
for coin in coinlist:
421
    time.sleep(1)
422
rez = get_pred(model, coin)
423
print('coin: ' + str(coin) + '; NN prediction: ' + str(rez))
424
if rez > 0.5:
425
    buyCoin = coin
426
trigger = 1
427
preds.append(rez)
428
break
429
​
430
if trigger == 1:
431
    BuyPrice = ticker(buyCoin)
432
CRYPTOBAL = openO(USDTBAL, buyCoin)
433
ID = limitClose(buyCoin, CRYPTOBAL * 0.999 //1, BuyPrice*1.00575)#traidng fees are added
434
        trigger = 2
435
​
436
        if trigger == 2:
437
        time.sleep(60 * 5) tick = ticker(buyCoin) check = tick / BuyPrice rez = get_pred(model, coin) check = checkOrder(buyCoin, ID) if check == True:
438
        USDTBAL = CRYPTOBAL * 0.999 * BuyPrice * 1.00575# traidng fees are added trigger = 0
439
        continue
440
        if rez < 0.5:
441
        USDTBAL = closeO(CRYPTOBAL * 0.999 //1,buyCoin)#traidng fees are added
442
            trigger = 0

Options:
HTML <style>, <script> formatting:
Enter your messy, minified, or obfuscated Python into the field above to have it cleaned up and made pretty. The editor above also contains helpful line numbers and syntax highlighting. There are many option to tailor the beautifier to your personal formatting tastes.

When do you use Python Viewer, Formatter, Editor
Often when writing Python your indentation, spacing, and other formatting can become a bit disorganized. It is also common for multiple developers to work on a single project who have different formatting techniques.  This tool is helpful for making the formatting of a file consistent. It is also common for Python to be minified or obfuscated. You can use this tool to make that code look pretty and readable so it is easier to edit.

Examples
The minified Python below:

if test == 1:
print 'it is one'
else:
print 'it is not one'
Becomes this beautified Python:

if test == 1:
    print 'it is one'
else :
    print 'it is not one'
DansTools.com
Contact
About
© 2014 - 2022 Dan's Tools
Contact Us Privacy Policy